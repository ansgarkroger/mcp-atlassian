"""Attachment operations for Jira API."""

import fnmatch
import logging
import mimetypes
import os
import re
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests

from ..models.jira import JiraAttachment
from ..utils.io import validate_safe_path
from ..utils.media import ATTACHMENT_MAX_BYTES
from .client import JiraClient
from .protocols import AttachmentsOperationsProto

# Configure logging
logger = logging.getLogger("mcp-jira")

# Jira Server/DC serves attachment bytes from /secure/attachment/{id}/{filename}
# and resolves the file by the id alone; the filename is decorative.
_SECURE_ATTACHMENT_URL = re.compile(
    r"^(?P<prefix>.*/secure/attachment/\d+)/(?P<name>[^/?#]*)(?P<suffix>[?#].*)?$"
)
_HTML_TITLE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_CHUNK_SIZE = 64 * 1024


class AttachmentFetchError(Exception):
    """An attachment could not be fetched; the message says why.

    Attributes:
        status: The HTTP status code when the server answered with one.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def attachment_url_candidates(url: str) -> list[str]:
    """Return ``url`` followed by filter-safe spellings of it (Server/DC).

    A URL filter in front of Jira Server/DC may deny a download because of the
    file extension in the path: an EC "Web Filter" was observed answering
    ``502 Access Denied`` for every ``*.log`` attachment while ``.txt``,
    ``.7z`` and ``.png`` files on the same issue downloaded fine. Jira resolves
    ``/secure/attachment/{id}/...`` by the id alone, so the same bytes are
    served when the extension's dot is percent-encoded or the filename is
    dropped, and a suffix filter no longer sees a blocked extension.

    Cloud content URLs carry no filename and are returned unchanged.

    Args:
        url: The attachment's ``content`` URL.

    Returns:
        The URLs to try, in order; the first one is ``url`` itself.
    """
    match = _SECURE_ATTACHMENT_URL.match(url)
    if not match:
        return [url]
    prefix = match.group("prefix")
    name = match.group("name")
    suffix = match.group("suffix") or ""
    candidates = [url]
    if "." in name:
        stem, _, extension = name.rpartition(".")
        candidates.append(f"{prefix}/{stem}%2E{extension}{suffix}")
    candidates.append(f"{prefix}/{suffix}")
    return candidates


def _matches_pattern(filename: str, pattern: str | None) -> bool:
    """Case-insensitive glob match of an attachment filename."""
    if not pattern:
        return True
    return fnmatch.fnmatchcase(filename.lower(), pattern.lower())


def _html_page_title(response: requests.Response) -> str:
    """Return the ``<title>`` of an HTML error page, or '' when there is none.

    A proxy or filter that refuses a download usually says why in a small
    HTML page; its title ("Web Filter", "Sign in", ...) is the one line that
    tells an operator what stood in the way.
    """
    try:
        content_type = response.headers.get("Content-Type", "")
        if not isinstance(content_type, str) or "html" not in content_type.lower():
            return ""
        head = next(iter(response.iter_content(chunk_size=4096)), b"")
        if not isinstance(head, bytes):
            return ""
        match = _HTML_TITLE.search(head.decode("utf-8", errors="replace"))
        return " ".join(match.group(1).split()) if match else ""
    except Exception:  # noqa: BLE001 - diagnostics must never mask the error
        return ""


class AttachmentsMixin(JiraClient, AttachmentsOperationsProto):
    """Mixin for Jira attachment operations."""

    # ------------------------------------------------------------------ transport

    def _open_attachment(self, url: str) -> requests.Response:
        """GET ``url`` as a stream.

        Args:
            url: The URL to fetch.

        Returns:
            The open, streaming response.

        Raises:
            AttachmentFetchError: On a transport error, an HTTP error status
                (with the HTML page title when the server sent one), or a
                redirect to a login page, which is what Jira Server/DC does
                when the credentials are not accepted for ``/secure/``.
        """
        try:
            response = self.jira._session.get(url, stream=True)
        except Exception as exc:  # noqa: BLE001 - includes the SSRF redirect hook
            raise AttachmentFetchError(f"{exc} for {url}") from exc

        status = response.status_code
        if not isinstance(status, int):
            status = None
        try:
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - requests raises HTTPError; mocks may not
            title = _html_page_title(response)
            response.close()
            reason = response.reason if isinstance(response.reason, str) else str(exc)
            detail = f' (page title: "{title}")' if title else ""
            raise AttachmentFetchError(
                f"HTTP {status if status is not None else '?'} {reason}{detail} for {url}",
                status=status,
            ) from exc

        history = response.history if isinstance(response.history, list) else []
        final_url = response.url if isinstance(response.url, str) else ""
        if history and "login" in final_url.lower():
            response.close()
            raise AttachmentFetchError(
                f"Jira redirected the download to a login page ({final_url}); "
                f"the configured credentials are not accepted for {url}"
            )
        return response

    def _open_attachment_with_fallback(self, url: str) -> requests.Response:
        """Open ``url``, retrying its filter-safe spellings after an HTTP error.

        Args:
            url: The attachment's ``content`` URL.

        Returns:
            The open, streaming response of the first spelling that worked.

        Raises:
            AttachmentFetchError: When every spelling failed; the message lists
                each attempt.
        """
        errors: list[str] = []
        for candidate in attachment_url_candidates(url):
            try:
                response = self._open_attachment(candidate)
            except AttachmentFetchError as exc:
                errors.append(str(exc))
                if exc.status is None:
                    # No HTTP answer to work around: another spelling of the
                    # same URL would only fail the same way.
                    break
                continue
            if candidate != url:
                logger.info(f"Fetched {url} through filter-safe URL {candidate}")
            return response
        raise AttachmentFetchError("; then ".join(errors))

    @staticmethod
    def _iter_response(response: requests.Response) -> Iterator[bytes]:
        """Yield the body of ``response`` in chunks and close it afterwards."""
        with response:
            yield from response.iter_content(chunk_size=8192)

    @staticmethod
    def _write_chunks(
        chunks: Iterator[bytes], target_path: str, expected_size: int | None = None
    ) -> int:
        """Write ``chunks`` to ``target_path`` and return the size written.

        Args:
            chunks: The file content.
            target_path: Absolute path of the file to create or overwrite.
            expected_size: The size Jira reports for the attachment; a
                mismatch is treated as a failed download and the file is
                removed, because a truncated log read as complete is worse
                than no log.

        Raises:
            AttachmentFetchError: If the file is missing afterwards or its
                size does not match ``expected_size``.
        """
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as f:
            for chunk in chunks:
                f.write(chunk)

        if not os.path.exists(target_path):
            raise AttachmentFetchError(f"File was not created at {target_path}")
        size = os.path.getsize(target_path)
        if expected_size and size != expected_size:
            Path(target_path).unlink(missing_ok=True)
            raise AttachmentFetchError(
                f"downloaded {size} bytes but Jira reports {expected_size} bytes "
                f"for {target_path}; the file was discarded"
            )
        return size

    # ------------------------------------------------------- single attachment

    def download_attachment(
        self, url: str, target_path: str, expected_size: int | None = None
    ) -> bool:
        """
        Download a Jira attachment to the specified path.

        Args:
            url: The URL of the attachment to download
            target_path: The path where the attachment should be saved
            expected_size: The size Jira reports for the attachment, if known;
                a download of a different size is discarded

        Returns:
            True if successful, False otherwise
        """
        if not url:
            logger.error("No URL provided for attachment download")
            return False

        try:
            # Convert to absolute path if relative
            if not os.path.isabs(target_path):
                target_path = os.path.abspath(target_path)

            # Guard against path traversal (resolves symlinks)
            validate_safe_path(target_path)

            # Do not write into the working-directory root itself: that is where
            # Python resolves imports first, so a download landing there could
            # overwrite an importable module and gain code execution (GHSA-6vmq).
            # Attachments must be saved into a subdirectory.
            resolved = Path(target_path).resolve()
            if resolved.parent == Path(os.getcwd()).resolve():
                logger.error(
                    f"Refusing to download into the working-directory root: {target_path}"
                )
                return False

            logger.info(f"Downloading attachment from {url} to {target_path}")

            response = self._open_attachment_with_fallback(url)
            file_size = self._write_chunks(
                self._iter_response(response), target_path, expected_size
            )
            logger.info(
                f"Successfully downloaded attachment to {target_path} (size: {file_size} bytes)"
            )
            return True

        except Exception as e:
            logger.error(f"Error downloading attachment: {str(e)}")
            return False

    def fetch_attachment_content(self, url: str) -> bytes | None:
        """
        Fetch attachment content into memory.

        Args:
            url: The URL of the attachment to download

        Returns:
            The raw bytes of the attachment, or None on failure
        """
        if not url:
            logger.error("No URL provided for attachment fetch")
            return None

        try:
            logger.info(f"Fetching attachment from {url}")
            response = self._open_attachment_with_fallback(url)
            data = b"".join(self._iter_response(response))
            logger.info(
                f"Successfully fetched attachment from {url} (size: {len(data)} bytes)"
            )
            return data

        except Exception as e:
            logger.error(f"Error fetching attachment: {str(e)}")
            return None

    # ------------------------------------------------- issue attachment archive

    def _issue_attachment_archive(
        self, issue_id: str, cache: dict[str, Any]
    ) -> zipfile.ZipFile:
        """Return the issue's *Download all* archive, fetched once per call.

        ``/secure/attachmentzip/{issue_id}.zip`` is the endpoint behind the
        issue view's *Download all* action on Jira Server/DC. It is the last
        resort when an attachment's own URL is refused: the archive carries
        every attachment under one ``.zip`` path, so a filter keyed on the
        extension of the blocked file does not apply to it.

        Args:
            issue_id: The numeric issue id (not the key).
            cache: Per-call state; the archive, or the reason it could not be
                fetched, is kept here so an issue with several blocked
                attachments downloads it once. Release it with
                :meth:`_close_archive`.

        Raises:
            AttachmentFetchError: If the archive cannot be fetched or read.
        """
        if "archive" in cache:
            return cache["archive"]
        if "error" in cache:
            raise AttachmentFetchError(cache["error"])

        url = f"{self.config.url.rstrip('/')}/secure/attachmentzip/{issue_id}.zip"
        spool = tempfile.TemporaryFile()
        try:
            response = self._open_attachment(url)
            for chunk in self._iter_response(response):
                spool.write(chunk)
            spool.seek(0)
            archive = zipfile.ZipFile(spool)
        except Exception as exc:  # noqa: BLE001 - transport, HTTP or a bad zip alike
            spool.close()
            cache["error"] = f"attachment archive {url}: {exc}"
            raise AttachmentFetchError(cache["error"]) from exc
        cache["archive"] = archive
        cache["spool"] = spool
        return archive

    @staticmethod
    def _close_archive(cache: dict[str, Any]) -> None:
        """Release the archive kept in ``cache``, if any."""
        archive = cache.pop("archive", None)
        if archive is not None:
            archive.close()
        spool = cache.pop("spool", None)
        if spool is not None:
            spool.close()

    @staticmethod
    def _archive_member(
        archive: zipfile.ZipFile, attachment: JiraAttachment
    ) -> zipfile.ZipInfo:
        """Locate ``attachment`` in the issue archive by name, then by size."""
        members = [
            info for info in archive.infolist() if info.filename == attachment.filename
        ]
        if len(members) > 1 and attachment.size:
            members = [
                info for info in members if info.file_size == attachment.size
            ] or members
        if not members:
            raise AttachmentFetchError(
                f"'{attachment.filename}' is not in the issue's attachment archive"
            )
        return members[0]

    @staticmethod
    def _iter_archive_member(
        archive: zipfile.ZipFile, member: zipfile.ZipInfo
    ) -> Iterator[bytes]:
        """Yield the content of one archive member in chunks."""
        with archive.open(member) as stream:
            while chunk := stream.read(_CHUNK_SIZE):
                yield chunk

    def _open_issue_attachment(
        self,
        attachment: JiraAttachment,
        issue_id: str | None,
        cache: dict[str, Any],
    ) -> Iterator[bytes]:
        """Yield the bytes of one attachment of an issue.

        The attachment's own URL (and its filter-safe spellings) is tried
        first. On Jira Server/DC the issue's *Download all* archive is the
        fallback, which is what a user does by hand when a single download is
        refused.

        Args:
            attachment: The attachment to fetch.
            issue_id: The numeric id of the issue, needed for the archive.
            cache: Per-call archive state, see :meth:`_issue_attachment_archive`.

        Raises:
            AttachmentFetchError: When every route failed; the message names
                each one.
        """
        try:
            response = self._open_attachment_with_fallback(attachment.url or "")
        except AttachmentFetchError as direct_error:
            if self.config.is_cloud or not issue_id:
                raise
            logger.warning(
                f"Direct download of {attachment.filename} failed ({direct_error}); "
                "falling back to the issue's attachment archive"
            )
            try:
                archive = self._issue_attachment_archive(issue_id, cache)
                member = self._archive_member(archive, attachment)
            except AttachmentFetchError as archive_error:
                raise AttachmentFetchError(
                    f"{direct_error}; archive fallback failed: {archive_error}"
                ) from archive_error
            return self._iter_archive_member(archive, member)
        return self._iter_response(response)

    # ------------------------------------------------------- issue attachments

    def get_issue_attachments(self, issue_key: str) -> list[JiraAttachment]:
        """Return attachment metadata for a Jira issue without downloading.

        Args:
            issue_key: The Jira issue key (e.g., 'PROJ-123').

        Returns:
            A list of JiraAttachment instances.
        """
        logger.info(f"Fetching attachment metadata for {issue_key}")
        issue_data = self.jira.issue(issue_key, fields="attachment")

        if not isinstance(issue_data, dict):
            msg = f"Unexpected return value type from `jira.issue`: {type(issue_data)}"
            logger.error(msg)
            raise TypeError(msg)

        if "fields" not in issue_data:
            logger.error(f"Could not retrieve issue {issue_key}")
            return []

        attachment_data = issue_data.get("fields", {}).get("attachment", [])
        return [
            JiraAttachment.from_api_response(item)
            for item in attachment_data
            if isinstance(item, dict)
        ]

    def get_issue_attachment_contents(
        self, issue_key: str, filename_pattern: str | None = None
    ) -> dict[str, Any]:
        """
        Fetch all attachment contents for a Jira issue into memory.

        Unlike download_issue_attachments, this method does not write to
        the filesystem.  Each attachment is returned as raw bytes so the
        caller (e.g. the MCP server layer) can serialise them however it
        needs (base64 embedded resources, etc.).

        Args:
            issue_key: The Jira issue key (e.g., 'PROJ-123')
            filename_pattern: Optional case-insensitive glob (``*.log``,
                ``report-??.pdf``); attachments whose filename does not match
                are skipped.

        Returns:
            A dictionary with:
                success (bool)
                issue_key (str)
                total (int): attachments on the issue, before filtering
                attachments (list[dict]): each dict has 'filename',
                    'content_type', 'size', and 'data' (bytes)
                failed (list[dict]): each dict has 'filename' and 'error'
                skipped (list[str]): filenames the pattern excluded
        """
        logger.info(f"Fetching attachment contents for {issue_key}")

        issue_data = self.jira.issue(issue_key, fields="attachment")

        if not isinstance(issue_data, dict):
            msg = f"Unexpected return value type from `jira.issue`: {type(issue_data)}"
            logger.error(msg)
            raise TypeError(msg)

        if "fields" not in issue_data:
            logger.error(f"Could not retrieve issue {issue_key}")
            return {
                "success": False,
                "error": f"Could not retrieve issue {issue_key}",
            }

        attachment_data = issue_data.get("fields", {}).get("attachment", [])
        issue_id = str(issue_data["id"]) if issue_data.get("id") else None

        attachments: list[JiraAttachment] = [
            JiraAttachment.from_api_response(item)
            for item in attachment_data
            if isinstance(item, dict)
        ]

        if not attachments:
            return {
                "success": True,
                "message": f"No attachments found for issue {issue_key}",
                "attachments": [],
                "failed": [],
                "skipped": [],
            }

        fetched: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        skipped: list[str] = []
        cache: dict[str, Any] = {}

        try:
            for attachment in attachments:
                if not _matches_pattern(attachment.filename, filename_pattern):
                    skipped.append(attachment.filename)
                    continue

                if not attachment.url:
                    logger.warning(f"No URL for attachment {attachment.filename}")
                    failed.append(
                        {"filename": attachment.filename, "error": "No URL available"}
                    )
                    continue

                if attachment.size > ATTACHMENT_MAX_BYTES:
                    logger.warning(
                        f"Skipping attachment {attachment.filename}: "
                        f"{attachment.size} bytes exceeds 50 MB limit"
                    )
                    failed.append(
                        {
                            "filename": attachment.filename,
                            "error": (
                                f"Attachment '{attachment.filename}' is "
                                f"{attachment.size} bytes which exceeds "
                                "the 50 MB inline limit. Retrieve it "
                                "directly from Jira."
                            ),
                        }
                    )
                    continue

                try:
                    data = b"".join(
                        self._open_issue_attachment(attachment, issue_id, cache)
                    )
                except Exception as exc:  # noqa: BLE001 - report, keep going
                    logger.error(
                        f"Error fetching attachment {attachment.filename}: {exc}"
                    )
                    failed.append(
                        {
                            "filename": attachment.filename,
                            "error": str(exc) or "Fetch failed",
                        }
                    )
                    continue

                content_type = (
                    attachment.content_type
                    or mimetypes.guess_type(attachment.filename)[0]
                    or "application/octet-stream"
                )
                fetched.append(
                    {
                        "filename": attachment.filename,
                        "content_type": content_type,
                        "size": len(data),
                        "data": data,
                    }
                )
        finally:
            self._close_archive(cache)

        return {
            "success": True,
            "issue_key": issue_key,
            "total": len(attachments),
            "attachments": fetched,
            "failed": failed,
            "skipped": skipped,
        }

    def download_issue_attachments(
        self,
        issue_key: str,
        target_dir: str,
        filename_pattern: str | None = None,
        base_dir: str | os.PathLike[str] | None = None,
    ) -> dict[str, Any]:
        """
        Download all attachments for a Jira issue.

        Args:
            issue_key: The Jira issue key (e.g., 'PROJ-123')
            target_dir: The directory where attachments should be saved
            filename_pattern: Optional case-insensitive glob (``*.log``);
                attachments whose filename does not match are skipped
            base_dir: The directory ``target_dir`` must stay within. A
                relative ``target_dir`` is resolved against it. Defaults to
                the current working directory.

        Returns:
            A dictionary with download results

        Raises:
            ValueError: If ``target_dir`` escapes ``base_dir`` or is
                ``base_dir`` itself.
        """
        if base_dir is not None:
            target_path = validate_safe_path(target_dir, base_dir=base_dir)
            if target_path == Path(base_dir).resolve(strict=False):
                raise ValueError(
                    f"target_dir must be a subdirectory of {base_dir}, "
                    "not the directory itself"
                )
        else:
            # Convert to absolute path if relative
            if not os.path.isabs(target_dir):
                target_dir = os.path.abspath(target_dir)

            # Guard against path traversal (resolves symlinks)
            validate_safe_path(target_dir)
            target_path = Path(target_dir)
            # Never write into the working-directory root (GHSA-6vmq).
            if target_path.resolve() == Path(os.getcwd()).resolve():
                raise ValueError(
                    "Refusing to download into the working-directory root; "
                    "choose a subdirectory"
                )

        logger.info(
            f"Downloading attachments for {issue_key} to directory: {target_path}"
        )

        # Create the target directory if it doesn't exist
        target_path.mkdir(parents=True, exist_ok=True)

        # Get the issue with attachments
        logger.info(f"Fetching issue {issue_key} with attachments")
        issue_data = self.jira.issue(issue_key, fields="attachment")

        if not isinstance(issue_data, dict):
            msg = f"Unexpected return value type from `jira.issue`: {type(issue_data)}"
            logger.error(msg)
            raise TypeError(msg)

        if "fields" not in issue_data:
            logger.error(f"Could not retrieve issue {issue_key}")
            return {"success": False, "error": f"Could not retrieve issue {issue_key}"}

        # Extract attachments from the API response
        attachment_data = issue_data.get("fields", {}).get("attachment", [])
        issue_id = str(issue_data["id"]) if issue_data.get("id") else None

        if not attachment_data:
            return {
                "success": True,
                "message": f"No attachments found for issue {issue_key}",
                "downloaded": [],
                "failed": [],
                "skipped": [],
            }

        # Create JiraAttachment objects for each attachment
        attachments = [
            JiraAttachment.from_api_response(item)
            for item in attachment_data
            if isinstance(item, dict)
        ]

        # Download each attachment
        downloaded: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        skipped: list[str] = []
        cache: dict[str, Any] = {}

        try:
            for attachment in attachments:
                if not _matches_pattern(attachment.filename, filename_pattern):
                    skipped.append(attachment.filename)
                    continue

                if not attachment.url:
                    logger.warning(f"No URL for attachment {attachment.filename}")
                    failed.append(
                        {"filename": attachment.filename, "error": "No URL available"}
                    )
                    continue

                # Create a safe filename
                safe_filename = Path(attachment.filename).name
                file_path = target_path / safe_filename

                try:
                    chunks = self._open_issue_attachment(attachment, issue_id, cache)
                    size = self._write_chunks(
                        chunks, str(file_path), attachment.size or None
                    )
                except Exception as exc:  # noqa: BLE001 - report, keep going
                    logger.error(
                        f"Error downloading attachment {attachment.filename}: {exc}"
                    )
                    failed.append(
                        {
                            "filename": attachment.filename,
                            "error": str(exc) or "Download failed",
                        }
                    )
                    continue

                logger.info(
                    f"Successfully downloaded attachment to {file_path} "
                    f"(size: {size} bytes)"
                )
                downloaded.append(
                    {
                        "filename": attachment.filename,
                        "path": str(file_path),
                        "size": size,
                    }
                )
        finally:
            self._close_archive(cache)

        return {
            "success": True,
            "issue_key": issue_key,
            "total": len(attachments),
            "downloaded": downloaded,
            "failed": failed,
            "skipped": skipped,
        }

    def upload_attachment(self, issue_key: str, file_path: str) -> dict[str, Any]:
        """
        Upload a single attachment to a Jira issue.

        Args:
            issue_key: The Jira issue key (e.g., 'PROJ-123')
            file_path: The path to the file to upload

        Returns:
            A dictionary with upload result information
        """
        if not issue_key:
            logger.error("No issue key provided for attachment upload")
            return {"success": False, "error": "No issue key provided"}

        if not file_path:
            logger.error("No file path provided for attachment upload")
            return {"success": False, "error": "No file path provided"}

        try:
            # Confine the upload source to the workspace before it is read: reject
            # traversal/absolute paths that escape CWD (arbitrary file read /
            # exfiltration via a caller-supplied file_path).
            file_path = str(validate_safe_path(file_path))

            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return {"success": False, "error": f"File not found: {file_path}"}

            logger.info(f"Uploading attachment from {file_path} to issue {issue_key}")

            # Use the Jira API to upload the file
            filename = os.path.basename(file_path)
            with open(file_path, "rb") as file:
                attachment = self.jira.add_attachment(
                    issue_key=issue_key, filename=file_path
                )

            if attachment:
                file_size = os.path.getsize(file_path)
                logger.info(
                    f"Successfully uploaded attachment {filename} to {issue_key} (size: {file_size} bytes)"
                )
                return {
                    "success": True,
                    "issue_key": issue_key,
                    "filename": filename,
                    "size": file_size,
                    "id": attachment.get("id")
                    if isinstance(attachment, dict)
                    else None,
                }
            else:
                logger.error(f"Failed to upload attachment {filename} to {issue_key}")
                return {
                    "success": False,
                    "error": f"Failed to upload attachment {filename} to {issue_key}",
                }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error uploading attachment: {error_msg}")
            return {"success": False, "error": error_msg}

    def upload_attachments(
        self, issue_key: str, file_paths: list[str]
    ) -> dict[str, Any]:
        """
        Upload multiple attachments to a Jira issue.

        Args:
            issue_key: The Jira issue key (e.g., 'PROJ-123')
            file_paths: List of paths to files to upload

        Returns:
            A dictionary with upload results
        """
        if not issue_key:
            logger.error("No issue key provided for attachment upload")
            return {"success": False, "error": "No issue key provided"}

        if not file_paths:
            logger.error("No file paths provided for attachment upload")
            return {"success": False, "error": "No file paths provided"}

        logger.info(f"Uploading {len(file_paths)} attachments to issue {issue_key}")

        # Upload each attachment
        uploaded = []
        failed = []

        for file_path in file_paths:
            result = self.upload_attachment(issue_key, file_path)

            if result.get("success"):
                uploaded.append(
                    {
                        "filename": result.get("filename"),
                        "size": result.get("size"),
                        "id": result.get("id"),
                    }
                )
            else:
                failed.append(
                    {
                        "filename": os.path.basename(file_path),
                        "error": result.get("error"),
                    }
                )

        return {
            "success": bool(uploaded),
            "issue_key": issue_key,
            "total": len(file_paths),
            "uploaded": uploaded,
            "failed": failed,
        }
