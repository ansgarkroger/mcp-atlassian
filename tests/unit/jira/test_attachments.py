"""Tests for the Jira attachments module."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from mcp_atlassian.jira import JiraFetcher
from mcp_atlassian.jira.attachments import (
    AttachmentFetchError,
    AttachmentsMixin,
    attachment_url_candidates,
)

# Test scenarios for AttachmentsMixin
#
# 1. Single Attachment Download (download_attachment method):
#    - Success case: Downloads attachment correctly with proper HTTP response
#    - Path handling: Converts relative path to absolute path
#    - Error cases:
#      - No URL provided
#      - HTTP error during download
#      - File write error
#      - File not created after write operation
#
# 2. Issue Attachments Download (download_issue_attachments method):
#    - Success case: Downloads all attachments for an issue
#    - Path handling: Converts relative target directory to absolute path
#    - Edge cases:
#      - Issue has no attachments
#      - Issue not found
#      - Issue has no fields
#      - Some attachments fail to download
#      - Attachment has missing URL
#
# 3. Single Attachment Upload (upload_attachment method):
#    - Success case: Uploads file correctly
#    - Path handling: Converts relative file path to absolute path
#    - Error cases:
#      - No issue key provided
#      - No file path provided
#      - File not found
#      - API error during upload
#      - No response from API
#
# 4. Multiple Attachments Upload (upload_attachments method):
#    - Success case: Uploads multiple files correctly
#    - Partial success: Some files upload successfully, others fail
#    - Error cases:
#      - Empty list of file paths
#      - No issue key provided


class TestAttachmentsMixin:
    """Tests for the AttachmentsMixin class."""

    @pytest.fixture
    def attachments_mixin(self, jira_fetcher: JiraFetcher) -> AttachmentsMixin:
        """Set up test fixtures before each test method."""
        # Create a mock Jira client
        attachments_mixin = jira_fetcher
        attachments_mixin.jira = MagicMock()
        attachments_mixin.jira._session = MagicMock()
        return attachments_mixin

    def test_download_attachment_success(self, attachments_mixin: AttachmentsMixin):
        """Test successful attachment download."""
        import os

        # Mock the response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_response.raise_for_status = MagicMock()
        attachments_mixin.jira._session.get.return_value = mock_response

        # A subdirectory of CWD (downloads may not land in the CWD root itself).
        download_path = "/tmp/downloads/test_file.txt"
        expected_path = os.path.abspath(download_path)

        # Mock file operations
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("os.path.exists") as mock_exists,
            patch("os.path.getsize") as mock_getsize,
            patch("os.makedirs") as mock_makedirs,
            patch("os.getcwd", return_value="/tmp"),
        ):
            mock_exists.return_value = True
            mock_getsize.return_value = 12  # Length of "test content"

            # Call the method
            result = attachments_mixin.download_attachment(
                "https://test.url/attachment", download_path
            )

            # Assertions
            assert result is True
            attachments_mixin.jira._session.get.assert_called_once_with(
                "https://test.url/attachment", stream=True
            )
            mock_file.assert_called_once_with(expected_path, "wb")
            mock_file().write.assert_called_once_with(b"test content")
            mock_makedirs.assert_called_once()

    def test_download_attachment_relative_path(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test attachment download with a relative path."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_response.raise_for_status = MagicMock()
        attachments_mixin.jira._session.get.return_value = mock_response

        # Mock file operations and os.path.abspath
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("os.path.exists") as mock_exists,
            patch("os.path.getsize") as mock_getsize,
            patch("os.makedirs") as mock_makedirs,
            patch("os.path.abspath") as mock_abspath,
            patch("os.path.isabs") as mock_isabs,
            patch("os.getcwd", return_value="/absolute/path"),
            patch("mcp_atlassian.jira.attachments.validate_safe_path"),
        ):
            mock_exists.return_value = True
            mock_getsize.return_value = 12
            mock_isabs.return_value = False
            # Target a subdirectory (downloads may not land in the CWD root itself).
            mock_abspath.side_effect = lambda p: (
                "/absolute/path/downloads/test_file.txt"
                if p == "downloads/test_file.txt"
                else p
            )

            # Call the method with a relative path
            result = attachments_mixin.download_attachment(
                "https://test.url/attachment", "downloads/test_file.txt"
            )

            # Assertions
            assert result is True
            mock_isabs.assert_any_call("downloads/test_file.txt")
            mock_abspath.assert_any_call("downloads/test_file.txt")
            mock_file.assert_called_once_with(
                "/absolute/path/downloads/test_file.txt", "wb"
            )

    def test_download_attachment_no_url(self, attachments_mixin: AttachmentsMixin):
        """Test attachment download with no URL."""
        result = attachments_mixin.download_attachment("", "/tmp/test_file.txt")
        assert result is False

    def test_download_attachment_http_error(self, attachments_mixin: AttachmentsMixin):
        """Test attachment download with an HTTP error."""
        # Mock the response to raise an HTTP error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        attachments_mixin.jira._session.get.return_value = mock_response

        result = attachments_mixin.download_attachment(
            "https://test.url/attachment", "/tmp/test_file.txt"
        )
        assert result is False

    def test_download_attachment_file_write_error(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test attachment download with a file write error."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_response.raise_for_status = MagicMock()
        attachments_mixin.jira._session.get.return_value = mock_response

        # Mock file operations to raise an exception during write
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("os.makedirs") as mock_makedirs,
        ):
            mock_file().write.side_effect = OSError("Write error")

            result = attachments_mixin.download_attachment(
                "https://test.url/attachment", "/tmp/test_file.txt"
            )
            assert result is False

    def test_download_attachment_file_not_created(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test attachment download when file is not created."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_response.raise_for_status = MagicMock()
        attachments_mixin.jira._session.get.return_value = mock_response

        # Mock file operations
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("os.path.exists") as mock_exists,
            patch("os.makedirs") as mock_makedirs,
        ):
            mock_exists.return_value = False  # File doesn't exist after write

            result = attachments_mixin.download_attachment(
                "https://test.url/attachment", "/tmp/test_file.txt"
            )
            assert result is False

    def test_download_issue_attachments_success(
        self, attachments_mixin: AttachmentsMixin, tmp_path: Path
    ):
        """Test successful download of all issue attachments."""
        # Mock the issue data
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "test1.txt",
                        "content": "https://test.url/attachment1",
                        "size": 100,
                    },
                    {
                        "filename": "test2.txt",
                        "content": "https://test.url/attachment2",
                        "size": 200,
                    },
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        # Mock JiraAttachment.from_api_response
        mock_attachment1 = MagicMock()
        mock_attachment1.filename = "test1.txt"
        mock_attachment1.url = "https://test.url/attachment1"
        mock_attachment1.size = 100

        mock_attachment2 = MagicMock()
        mock_attachment2.filename = "test2.txt"
        mock_attachment2.url = "https://test.url/attachment2"
        mock_attachment2.size = 200

        target = tmp_path / "attachments"
        with (
            patch.object(
                attachments_mixin,
                "_open_issue_attachment",
                side_effect=[iter([b"x" * 100]), iter([b"y" * 200])],
            ) as mock_download,
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                side_effect=[mock_attachment1, mock_attachment2],
            ),
            patch("os.getcwd", return_value=str(tmp_path)),
        ):
            result = attachments_mixin.download_issue_attachments(
                "TEST-123", str(target)
            )

            # Assertions
            assert result["success"] is True
            assert len(result["downloaded"]) == 2
            assert len(result["failed"]) == 0
            assert result["total"] == 2
            assert result["issue_key"] == "TEST-123"
            assert mock_download.call_count == 2
            assert (target / "test1.txt").stat().st_size == 100
            assert (target / "test2.txt").stat().st_size == 200
            assert result["downloaded"][0]["size"] == 100

    def test_download_issue_attachments_relative_path(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test download issue attachments with a relative path."""
        # Mock the issue data
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "test1.txt",
                        "content": "https://test.url/attachment1",
                        "size": 100,
                    }
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        # Mock attachment
        mock_attachment = MagicMock()
        mock_attachment.filename = "test1.txt"
        mock_attachment.url = "https://test.url/attachment1"
        mock_attachment.size = 100

        # Mock path operations
        with (
            patch.object(
                attachments_mixin, "_open_issue_attachment", return_value=iter([])
            ),
            patch.object(attachments_mixin, "_write_chunks", return_value=100),
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                return_value=mock_attachment,
            ),
            patch("os.path.isabs") as mock_isabs,
            patch("os.path.abspath") as mock_abspath,
            patch("os.getcwd", return_value="/absolute/path"),
            patch("mcp_atlassian.jira.attachments.validate_safe_path"),
        ):
            # Only the caller's path is relative; Path.resolve() must still
            # see the mocked absolute results as absolute.
            mock_isabs.side_effect = lambda p: str(p) != "attachments"
            mock_abspath.side_effect = lambda p: (
                "/absolute/path/attachments" if p == "attachments" else p
            )

            result = attachments_mixin.download_issue_attachments(
                "TEST-123", "attachments"
            )

            # Assertions
            assert result["success"] is True
            mock_isabs.assert_any_call("attachments")
            mock_abspath.assert_any_call("attachments")

    def test_download_issue_attachments_no_attachments(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test download when issue has no attachments."""
        # Mock the issue data with no attachments
        mock_issue = {"fields": {"attachment": []}}
        attachments_mixin.jira.issue.return_value = mock_issue

        with (
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch("os.getcwd", return_value="/tmp"),
        ):
            result = attachments_mixin.download_issue_attachments(
                "TEST-123", "/tmp/attachments"
            )

            # Assertions
            assert result["success"] is True
            assert "No attachments found" in result["message"]
            assert len(result["downloaded"]) == 0
            assert len(result["failed"]) == 0
            mock_mkdir.assert_called_once()

    def test_download_issue_attachments_issue_not_found(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test download when issue cannot be retrieved."""
        attachments_mixin.jira.issue.return_value = None

        with (
            patch("os.getcwd", return_value="/tmp"),
            pytest.raises(
                TypeError,
                match="Unexpected return value type from `jira.issue`: <class 'NoneType'>",
            ),
        ):
            attachments_mixin.download_issue_attachments("TEST-123", "/tmp/attachments")

    def test_download_issue_attachments_no_fields(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test download when issue has no fields."""
        # Mock the issue data with no fields
        mock_issue = {}  # Missing 'fields' key
        attachments_mixin.jira.issue.return_value = mock_issue

        with patch("os.getcwd", return_value="/tmp"):
            result = attachments_mixin.download_issue_attachments(
                "TEST-123", "/tmp/attachments"
            )

        # Assertions
        assert result["success"] is False
        assert "Could not retrieve issue" in result["error"]

    def test_download_issue_attachments_some_failures(
        self, attachments_mixin: AttachmentsMixin, tmp_path: Path
    ):
        """Test download when some attachments fail to download."""
        # Mock the issue data
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "test1.txt",
                        "content": "https://test.url/attachment1",
                        "size": 100,
                    },
                    {
                        "filename": "test2.txt",
                        "content": "https://test.url/attachment2",
                        "size": 200,
                    },
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        # Mock attachments
        mock_attachment1 = MagicMock()
        mock_attachment1.filename = "test1.txt"
        mock_attachment1.url = "https://test.url/attachment1"
        mock_attachment1.size = 100

        mock_attachment2 = MagicMock()
        mock_attachment2.filename = "test2.txt"
        mock_attachment2.url = "https://test.url/attachment2"
        mock_attachment2.size = 200

        # First attachment streams, the second is refused by the server
        with (
            patch.object(
                attachments_mixin,
                "_open_issue_attachment",
                side_effect=[
                    iter([b"x" * 100]),
                    AttachmentFetchError("HTTP 502 Bad Gateway for attachment2"),
                ],
            ) as mock_download,
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                side_effect=[mock_attachment1, mock_attachment2],
            ),
            patch("os.getcwd", return_value=str(tmp_path)),
        ):
            result = attachments_mixin.download_issue_attachments(
                "TEST-123", str(tmp_path / "attachments")
            )

            # Assertions
            assert result["success"] is True
            assert len(result["downloaded"]) == 1
            assert len(result["failed"]) == 1
            assert result["downloaded"][0]["filename"] == "test1.txt"
            assert result["failed"][0]["filename"] == "test2.txt"
            assert "502" in result["failed"][0]["error"]
            assert mock_download.call_count == 2

    def test_download_issue_attachments_missing_url(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test download when an attachment has no URL."""
        # Mock the issue data
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "test1.txt",
                        "content": "https://test.url/attachment1",
                        "size": 100,
                    }
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        # Mock attachment with no URL
        mock_attachment = MagicMock()
        mock_attachment.filename = "test1.txt"
        mock_attachment.url = None  # No URL
        mock_attachment.size = 100

        # Mock path operations
        with (
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                return_value=mock_attachment,
            ),
            patch("os.getcwd", return_value="/tmp"),
        ):
            result = attachments_mixin.download_issue_attachments(
                "TEST-123", "/tmp/attachments"
            )

            # Assertions
            assert result["success"] is True
            assert len(result["downloaded"]) == 0
            assert len(result["failed"]) == 1
            assert result["failed"][0]["filename"] == "test1.txt"
            assert "No URL available" in result["failed"][0]["error"]

    # Tests for upload_attachment method

    def test_upload_attachment_success(self, attachments_mixin: AttachmentsMixin):
        """Test successful attachment upload."""
        # Mock the Jira API response
        mock_attachment_response = {
            "id": "12345",
            "filename": "test_file.txt",
            "size": 100,
        }
        attachments_mixin.jira.add_attachment.return_value = mock_attachment_response

        # Mock file operations
        with (
            # Pin the workspace so the absolute path resolves inside it — keeps
            # validate_safe_path passing deterministically across Python versions
            # (mocking os.path.abspath does not reach pathlib.Path.resolve on 3.13).
            patch("os.getcwd", return_value="/absolute/path"),
            patch("os.path.exists") as mock_exists,
            patch("os.path.getsize") as mock_getsize,
            patch("os.path.isabs") as mock_isabs,
            patch("os.path.abspath") as mock_abspath,
            patch("os.path.basename") as mock_basename,
            patch("builtins.open", mock_open(read_data=b"test content")),
        ):
            mock_exists.return_value = True
            mock_getsize.return_value = 100
            mock_isabs.return_value = True
            mock_abspath.return_value = "/absolute/path/test_file.txt"
            mock_basename.return_value = "test_file.txt"

            # Call the method
            result = attachments_mixin.upload_attachment(
                "TEST-123", "/absolute/path/test_file.txt"
            )

            # Assertions
            assert result["success"] is True
            assert result["issue_key"] == "TEST-123"
            assert result["filename"] == "test_file.txt"
            assert result["size"] == 100
            assert result["id"] == "12345"
            attachments_mixin.jira.add_attachment.assert_called_once_with(
                issue_key="TEST-123", filename="/absolute/path/test_file.txt"
            )

    def test_upload_attachment_relative_path(
        self, attachments_mixin: AttachmentsMixin, tmp_path: Path
    ):
        """A relative path inside the workspace resolves and uploads."""
        mock_attachment_response = {
            "id": "12345",
            "filename": "test_file.txt",
            "size": 100,
        }
        attachments_mixin.jira.add_attachment.return_value = mock_attachment_response

        (tmp_path / "test_file.txt").write_bytes(b"test content")
        resolved = str((tmp_path / "test_file.txt").resolve())

        with patch("os.getcwd", return_value=str(tmp_path)):
            result = attachments_mixin.upload_attachment("TEST-123", "test_file.txt")

        assert result["success"] is True
        attachments_mixin.jira.add_attachment.assert_called_once_with(
            issue_key="TEST-123", filename=resolved
        )

    def test_upload_attachment_no_issue_key(self, attachments_mixin: AttachmentsMixin):
        """Test attachment upload with no issue key."""
        result = attachments_mixin.upload_attachment("", "/path/to/file.txt")

        # Assertions
        assert result["success"] is False
        assert "No issue key provided" in result["error"]
        attachments_mixin.jira.add_attachment.assert_not_called()

    def test_upload_attachment_no_file_path(self, attachments_mixin: AttachmentsMixin):
        """Test attachment upload with no file path."""
        result = attachments_mixin.upload_attachment("TEST-123", "")

        # Assertions
        assert result["success"] is False
        assert "No file path provided" in result["error"]
        attachments_mixin.jira.add_attachment.assert_not_called()

    def test_upload_attachment_file_not_found(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test attachment upload when file doesn't exist."""
        # Mock file operations
        with (
            patch("os.getcwd", return_value="/absolute/path"),
            patch("os.path.exists") as mock_exists,
            patch("os.path.isabs") as mock_isabs,
            patch("os.path.abspath") as mock_abspath,
            patch("builtins.open", mock_open(read_data=b"test content")),
        ):
            mock_exists.return_value = False
            mock_isabs.return_value = True
            mock_abspath.return_value = "/absolute/path/test_file.txt"

            result = attachments_mixin.upload_attachment(
                "TEST-123", "/absolute/path/test_file.txt"
            )

            # Assertions
            assert result["success"] is False
            assert "File not found" in result["error"]
            attachments_mixin.jira.add_attachment.assert_not_called()

    def test_upload_attachment_api_error(self, attachments_mixin: AttachmentsMixin):
        """Test attachment upload with an API error."""
        # Mock the Jira API to raise an exception
        attachments_mixin.jira.add_attachment.side_effect = Exception("API Error")

        # Mock file operations
        with (
            patch("os.getcwd", return_value="/absolute/path"),
            patch("os.path.exists") as mock_exists,
            patch("os.path.isabs") as mock_isabs,
            patch("os.path.abspath") as mock_abspath,
            patch("os.path.basename") as mock_basename,
            patch("builtins.open", mock_open(read_data=b"test content")),
        ):
            mock_exists.return_value = True
            mock_isabs.return_value = True
            mock_abspath.return_value = "/absolute/path/test_file.txt"
            mock_basename.return_value = "test_file.txt"

            result = attachments_mixin.upload_attachment(
                "TEST-123", "/absolute/path/test_file.txt"
            )

            # Assertions
            assert result["success"] is False
            assert "API Error" in result["error"]

    def test_upload_attachment_no_response(self, attachments_mixin: AttachmentsMixin):
        """Test attachment upload when API returns no response."""
        # Mock the Jira API to return None
        attachments_mixin.jira.add_attachment.return_value = None

        # Mock file operations
        with (
            patch("os.getcwd", return_value="/absolute/path"),
            patch("os.path.exists") as mock_exists,
            patch("os.path.isabs") as mock_isabs,
            patch("os.path.abspath") as mock_abspath,
            patch("os.path.basename") as mock_basename,
            patch("builtins.open", mock_open(read_data=b"test content")),
        ):
            mock_exists.return_value = True
            mock_isabs.return_value = True
            mock_abspath.return_value = "/absolute/path/test_file.txt"
            mock_basename.return_value = "test_file.txt"

            result = attachments_mixin.upload_attachment(
                "TEST-123", "/absolute/path/test_file.txt"
            )

            # Assertions
            assert result["success"] is False
            assert "Failed to upload attachment" in result["error"]

    # Tests for upload_attachments method

    def test_upload_attachments_success(self, attachments_mixin: AttachmentsMixin):
        """Test successful upload of multiple attachments."""
        # Set up mock for upload_attachment method to simulate successful uploads
        file_paths = [
            "/path/to/file1.txt",
            "/path/to/file2.pdf",
            "/path/to/file3.jpg",
        ]

        # Create mock successful results for each file
        mock_results = [
            {
                "success": True,
                "issue_key": "TEST-123",
                "filename": f"file{i + 1}.{ext}",
                "size": 100 * (i + 1),
                "id": f"id{i + 1}",
            }
            for i, ext in enumerate(["txt", "pdf", "jpg"])
        ]

        with patch.object(
            attachments_mixin, "upload_attachment", side_effect=mock_results
        ) as mock_upload:
            # Call the method
            result = attachments_mixin.upload_attachments("TEST-123", file_paths)

            # Assertions
            assert result["success"] is True
            assert result["issue_key"] == "TEST-123"
            assert result["total"] == 3
            assert len(result["uploaded"]) == 3
            assert len(result["failed"]) == 0

            # Check that upload_attachment was called for each file
            assert mock_upload.call_count == 3
            mock_upload.assert_any_call("TEST-123", "/path/to/file1.txt")
            mock_upload.assert_any_call("TEST-123", "/path/to/file2.pdf")
            mock_upload.assert_any_call("TEST-123", "/path/to/file3.jpg")

            # Verify uploaded files details
            assert result["uploaded"][0]["filename"] == "file1.txt"
            assert result["uploaded"][1]["filename"] == "file2.pdf"
            assert result["uploaded"][2]["filename"] == "file3.jpg"
            assert result["uploaded"][0]["size"] == 100
            assert result["uploaded"][1]["size"] == 200
            assert result["uploaded"][2]["size"] == 300
            assert result["uploaded"][0]["id"] == "id1"
            assert result["uploaded"][1]["id"] == "id2"
            assert result["uploaded"][2]["id"] == "id3"

    def test_upload_attachments_mixed_results(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test upload of multiple attachments with mixed success and failure."""
        # Set up mock for upload_attachment method to simulate mixed results
        file_paths = [
            "/path/to/file1.txt",  # Will succeed
            "/path/to/file2.pdf",  # Will fail
            "/path/to/file3.jpg",  # Will succeed
        ]

        # Create mock results with mixed success/failure
        mock_results = [
            {
                "success": True,
                "issue_key": "TEST-123",
                "filename": "file1.txt",
                "size": 100,
                "id": "id1",
            },
            {"success": False, "error": "File not found: /path/to/file2.pdf"},
            {
                "success": True,
                "issue_key": "TEST-123",
                "filename": "file3.jpg",
                "size": 300,
                "id": "id3",
            },
        ]

        with patch.object(
            attachments_mixin, "upload_attachment", side_effect=mock_results
        ) as mock_upload:
            # Call the method
            result = attachments_mixin.upload_attachments("TEST-123", file_paths)

            # Assertions
            assert (
                result["success"] is True
            )  # Overall success is True even with partial failures
            assert result["issue_key"] == "TEST-123"
            assert result["total"] == 3
            assert len(result["uploaded"]) == 2
            assert len(result["failed"]) == 1

            # Check that upload_attachment was called for each file
            assert mock_upload.call_count == 3

            # Verify uploaded files details
            assert result["uploaded"][0]["filename"] == "file1.txt"
            assert result["uploaded"][1]["filename"] == "file3.jpg"
            assert result["uploaded"][0]["size"] == 100
            assert result["uploaded"][1]["size"] == 300
            assert result["uploaded"][0]["id"] == "id1"
            assert result["uploaded"][1]["id"] == "id3"

            # Verify failed file details
            assert result["failed"][0]["filename"] == "file2.pdf"
            assert "File not found" in result["failed"][0]["error"]

    def test_upload_attachments_all_fail(self, attachments_mixin: AttachmentsMixin):
        """Report failure when no attachment is uploaded."""
        with patch.object(
            attachments_mixin,
            "upload_attachment",
            return_value={"success": False, "error": "File not found"},
        ):
            result = attachments_mixin.upload_attachments(
                "TEST-123", ["/path/to/file1.txt", "/path/to/file2.txt"]
            )

        assert result["success"] is False
        assert result["uploaded"] == []
        assert len(result["failed"]) == 2

    def test_upload_attachments_empty_list(self, attachments_mixin: AttachmentsMixin):
        """Test upload with an empty list of file paths."""
        # Call the method with an empty list
        result = attachments_mixin.upload_attachments("TEST-123", [])

        # Assertions
        assert result["success"] is False
        assert "No file paths provided" in result["error"]

    def test_upload_attachments_no_issue_key(self, attachments_mixin: AttachmentsMixin):
        """Test upload with no issue key provided."""
        # Call the method with no issue key
        result = attachments_mixin.upload_attachments("", ["/path/to/file.txt"])

        # Assertions
        assert result["success"] is False
        assert "No issue key provided" in result["error"]

    # Tests for fetch_attachment_content method

    def test_fetch_attachment_content_success(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test successful in-memory attachment fetch."""
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.raise_for_status = MagicMock()
        attachments_mixin.jira._session.get.return_value = mock_response

        result = attachments_mixin.fetch_attachment_content(
            "https://test.url/attachment"
        )

        assert result == b"chunk1chunk2"
        attachments_mixin.jira._session.get.assert_called_once_with(
            "https://test.url/attachment", stream=True
        )

    def test_fetch_attachment_content_no_url(self, attachments_mixin: AttachmentsMixin):
        """Test fetch with no URL returns None."""
        result = attachments_mixin.fetch_attachment_content("")
        assert result is None

    def test_fetch_attachment_content_http_error(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test fetch with an HTTP error returns None."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        attachments_mixin.jira._session.get.return_value = mock_response

        result = attachments_mixin.fetch_attachment_content(
            "https://test.url/attachment"
        )
        assert result is None

    # Tests for get_issue_attachment_contents method

    def test_get_issue_attachment_contents_success(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test successful in-memory fetch of all issue attachments."""
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "test1.txt",
                        "content": "https://test.url/attachment1",
                        "size": 100,
                        "mimeType": "text/plain",
                    },
                    {
                        "filename": "test2.png",
                        "content": "https://test.url/attachment2",
                        "size": 200,
                        "mimeType": "image/png",
                    },
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        mock_attachment1 = MagicMock()
        mock_attachment1.filename = "test1.txt"
        mock_attachment1.url = "https://test.url/attachment1"
        mock_attachment1.content_type = "text/plain"
        mock_attachment1.size = 100

        mock_attachment2 = MagicMock()
        mock_attachment2.filename = "test2.png"
        mock_attachment2.url = "https://test.url/attachment2"
        mock_attachment2.content_type = "image/png"
        mock_attachment2.size = 200

        with (
            patch.object(
                attachments_mixin,
                "_open_issue_attachment",
                side_effect=[iter([b"content1"]), iter([b"image_data"])],
            ) as mock_fetch,
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                side_effect=[mock_attachment1, mock_attachment2],
            ),
        ):
            result = attachments_mixin.get_issue_attachment_contents("TEST-123")

            assert result["success"] is True
            assert result["issue_key"] == "TEST-123"
            assert result["total"] == 2
            assert len(result["attachments"]) == 2
            assert len(result["failed"]) == 0
            assert result["attachments"][0]["filename"] == "test1.txt"
            assert result["attachments"][0]["data"] == b"content1"
            assert result["attachments"][0]["content_type"] == "text/plain"
            assert result["attachments"][1]["filename"] == "test2.png"
            assert result["attachments"][1]["data"] == b"image_data"
            assert mock_fetch.call_count == 2

    def test_get_issue_attachment_contents_no_attachments(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test fetch when issue has no attachments."""
        mock_issue = {"fields": {"attachment": []}}
        attachments_mixin.jira.issue.return_value = mock_issue

        result = attachments_mixin.get_issue_attachment_contents("TEST-123")

        assert result["success"] is True
        assert "No attachments found" in result["message"]
        assert len(result["attachments"]) == 0
        assert len(result["failed"]) == 0

    def test_get_issue_attachment_contents_issue_not_found(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test fetch when issue returns unexpected type."""
        attachments_mixin.jira.issue.return_value = None

        with pytest.raises(
            TypeError,
            match="Unexpected return value type from `jira.issue`: <class 'NoneType'>",
        ):
            attachments_mixin.get_issue_attachment_contents("TEST-123")

    def test_get_issue_attachment_contents_no_fields(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test fetch when issue has no fields."""
        mock_issue = {}
        attachments_mixin.jira.issue.return_value = mock_issue

        result = attachments_mixin.get_issue_attachment_contents("TEST-123")

        assert result["success"] is False
        assert "Could not retrieve issue" in result["error"]

    def test_get_issue_attachment_contents_some_failures(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test fetch when some attachments fail."""
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "good.txt",
                        "content": "https://test.url/good",
                        "size": 100,
                    },
                    {
                        "filename": "bad.txt",
                        "content": "https://test.url/bad",
                        "size": 200,
                    },
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        mock_attachment1 = MagicMock()
        mock_attachment1.filename = "good.txt"
        mock_attachment1.url = "https://test.url/good"
        mock_attachment1.content_type = "text/plain"
        mock_attachment1.size = 100

        mock_attachment2 = MagicMock()
        mock_attachment2.filename = "bad.txt"
        mock_attachment2.url = "https://test.url/bad"
        mock_attachment2.content_type = "text/plain"
        mock_attachment2.size = 200

        with (
            patch.object(
                attachments_mixin,
                "_open_issue_attachment",
                side_effect=[
                    iter([b"good_data"]),
                    AttachmentFetchError(
                        "HTTP 502 Bad Gateway for https://test.url/bad"
                    ),
                ],
            ),
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                side_effect=[mock_attachment1, mock_attachment2],
            ),
        ):
            result = attachments_mixin.get_issue_attachment_contents("TEST-123")

            assert result["success"] is True
            assert len(result["attachments"]) == 1
            assert len(result["failed"]) == 1
            assert result["attachments"][0]["filename"] == "good.txt"
            assert result["failed"][0]["filename"] == "bad.txt"
            assert "502" in result["failed"][0]["error"]

    def test_get_issue_attachment_contents_missing_url(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test fetch when an attachment has no URL."""
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "no_url.txt",
                        "size": 100,
                    }
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        mock_attachment = MagicMock()
        mock_attachment.filename = "no_url.txt"
        mock_attachment.url = None
        mock_attachment.content_type = None

        with patch(
            "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
            return_value=mock_attachment,
        ):
            result = attachments_mixin.get_issue_attachment_contents("TEST-123")

            assert result["success"] is True
            assert len(result["attachments"]) == 0
            assert len(result["failed"]) == 1
            assert "No URL available" in result["failed"][0]["error"]

    # Tests for 50MB attachment size limit

    def test_get_issue_attachment_contents_skips_oversized(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Attachment with size > 50MB in metadata is skipped before download."""
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "huge.bin",
                        "content": "https://test.url/huge",
                        "size": 60 * 1024 * 1024,  # 60 MB
                    }
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        mock_attachment = MagicMock()
        mock_attachment.filename = "huge.bin"
        mock_attachment.url = "https://test.url/huge"
        mock_attachment.size = 60 * 1024 * 1024
        mock_attachment.content_type = "application/octet-stream"

        with (
            patch.object(
                attachments_mixin,
                "_open_issue_attachment",
            ) as mock_fetch,
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                return_value=mock_attachment,
            ),
        ):
            result = attachments_mixin.get_issue_attachment_contents("TEST-123")

            # fetch_attachment_content should NOT be called for oversized file
            mock_fetch.assert_not_called()
            assert len(result["attachments"]) == 0
            assert len(result["failed"]) == 1
            assert "50 MB" in result["failed"][0]["error"]

    def test_get_issue_attachment_contents_allows_zero_size(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Attachment with size=0 (metadata missing) still gets downloaded."""
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "unknown.bin",
                        "content": "https://test.url/unknown",
                    }
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        mock_attachment = MagicMock()
        mock_attachment.filename = "unknown.bin"
        mock_attachment.url = "https://test.url/unknown"
        mock_attachment.size = 0
        mock_attachment.content_type = "application/octet-stream"

        with (
            patch.object(
                attachments_mixin,
                "_open_issue_attachment",
                return_value=iter([b"data"]),
            ) as mock_fetch,
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                return_value=mock_attachment,
            ),
        ):
            result = attachments_mixin.get_issue_attachment_contents("TEST-123")

            mock_fetch.assert_called_once()
            assert len(result["attachments"]) == 1
            assert len(result["failed"]) == 0

    def test_get_issue_attachment_contents_allows_small(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Attachment with size < 50MB gets downloaded normally."""
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "small.txt",
                        "content": "https://test.url/small",
                        "size": 1024,
                    }
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        mock_attachment = MagicMock()
        mock_attachment.filename = "small.txt"
        mock_attachment.url = "https://test.url/small"
        mock_attachment.size = 1024
        mock_attachment.content_type = "text/plain"

        with (
            patch.object(
                attachments_mixin,
                "_open_issue_attachment",
                return_value=iter([b"small content"]),
            ) as mock_fetch,
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                return_value=mock_attachment,
            ),
        ):
            result = attachments_mixin.get_issue_attachment_contents("TEST-123")

            mock_fetch.assert_called_once()
            assert len(result["attachments"]) == 1
            assert len(result["failed"]) == 0

    def test_get_issue_attachment_contents_mixed_sizes(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Mix of large and small attachments — only large ones skipped."""
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "small.txt",
                        "content": "https://test.url/small",
                        "size": 1024,
                    },
                    {
                        "filename": "huge.bin",
                        "content": "https://test.url/huge",
                        "size": 60 * 1024 * 1024,
                    },
                    {
                        "filename": "medium.txt",
                        "content": "https://test.url/medium",
                        "size": 10 * 1024 * 1024,
                    },
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        mock_small = MagicMock()
        mock_small.filename = "small.txt"
        mock_small.url = "https://test.url/small"
        mock_small.size = 1024
        mock_small.content_type = "text/plain"

        mock_huge = MagicMock()
        mock_huge.filename = "huge.bin"
        mock_huge.url = "https://test.url/huge"
        mock_huge.size = 60 * 1024 * 1024
        mock_huge.content_type = "application/octet-stream"

        mock_medium = MagicMock()
        mock_medium.filename = "medium.txt"
        mock_medium.url = "https://test.url/medium"
        mock_medium.size = 10 * 1024 * 1024
        mock_medium.content_type = "text/plain"

        with (
            patch.object(
                attachments_mixin,
                "_open_issue_attachment",
                side_effect=[iter([b"small data"]), iter([b"medium data"])],
            ) as mock_fetch,
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                side_effect=[mock_small, mock_huge, mock_medium],
            ),
        ):
            result = attachments_mixin.get_issue_attachment_contents("TEST-123")

            # Only small and medium should be fetched
            assert mock_fetch.call_count == 2
            assert len(result["attachments"]) == 2
            assert len(result["failed"]) == 1
            assert result["failed"][0]["filename"] == "huge.bin"
            assert "50 MB" in result["failed"][0]["error"]
            filenames = [a["filename"] for a in result["attachments"]]
            assert "small.txt" in filenames
            assert "medium.txt" in filenames

    # Tests for path traversal rejection (PR #949 security hardening)

    @pytest.mark.parametrize(
        "malicious_path",
        [
            # Absolute paths that resolve outside cwd
            "/etc/passwd",
            "/tmp/evil/file.txt",
            "/var/log/secrets.txt",
        ],
        ids=[
            "absolute-etc-passwd",
            "absolute-tmp-evil",
            "absolute-var-log",
        ],
    )
    def test_download_attachment_rejects_path_traversal(
        self,
        attachments_mixin: AttachmentsMixin,
        malicious_path: str,
    ):
        """Regression: path traversal in attachment filenames must be rejected.

        The guard in download_attachment checks that the resolved absolute path
        is within cwd via ``Path(target_path).is_relative_to(cwd)``.  Paths
        outside cwd trigger a ValueError which is caught internally, causing
        the method to return False.
        """
        with patch("os.getcwd", return_value="/safe/working/dir"):
            result = attachments_mixin.download_attachment(
                "https://test.url/attachment",
                malicious_path,
            )
            assert result is False

    @pytest.mark.parametrize(
        "relative_path",
        [
            "../../../etc/passwd",
            "normal/../../../etc/shadow",
        ],
        ids=[
            "relative-traversal-etc-passwd",
            "relative-traversal-nested",
        ],
    )
    def test_download_attachment_rejects_relative_path_traversal(
        self,
        attachments_mixin: AttachmentsMixin,
        relative_path: str,
    ):
        """Regression: relative paths that resolve outside cwd are rejected.

        When a relative path is given, os.path.abspath resolves it.  If the
        resolved path escapes cwd, the guard rejects it and returns False.
        """
        import os

        cwd = os.getcwd()
        with patch("os.getcwd", return_value=cwd):
            result = attachments_mixin.download_attachment(
                "https://test.url/attachment",
                relative_path,
            )
            # The relative path resolves outside cwd (it traverses up)
            resolved = os.path.abspath(relative_path)
            if not Path(resolved).is_relative_to(cwd):
                assert result is False

    @pytest.mark.parametrize(
        "malicious_dir",
        [
            "/etc",
            "/tmp/evil",
            "/var/log",
        ],
        ids=[
            "absolute-etc",
            "absolute-tmp-evil",
            "absolute-var-log",
        ],
    )
    def test_download_issue_attachments_rejects_path_traversal(
        self,
        attachments_mixin: AttachmentsMixin,
        malicious_dir: str,
    ):
        """Regression: path traversal in target_dir must raise ValueError.

        Unlike download_attachment (which catches the error internally),
        download_issue_attachments lets the ValueError propagate.
        """
        with (
            patch("os.getcwd", return_value="/safe/working/dir"),
            pytest.raises(ValueError, match="Path traversal detected"),
        ):
            attachments_mixin.download_issue_attachments(
                "TEST-123",
                malicious_dir,
            )

    def test_get_issue_attachments_metadata(self, attachments_mixin: AttachmentsMixin):
        """get_issue_attachments returns JiraAttachment list without downloading."""
        attachments_mixin.jira.issue.return_value = {
            "fields": {
                "attachment": [
                    {
                        "id": "101",
                        "filename": "photo.png",
                        "size": 1024,
                        "mimeType": "image/png",
                        "content": "https://jira.example.com/att/101",
                    },
                    {
                        "id": "102",
                        "filename": "report.pdf",
                        "size": 2048,
                        "mimeType": "application/pdf",
                        "content": "https://jira.example.com/att/102",
                    },
                ]
            }
        }

        from mcp_atlassian.models.jira import JiraAttachment

        result = attachments_mixin.get_issue_attachments("TEST-123")
        assert len(result) == 2
        assert all(isinstance(a, JiraAttachment) for a in result)
        assert result[0].filename == "photo.png"
        assert result[0].content_type == "image/png"
        assert result[1].filename == "report.pdf"
        # No download calls should have been made
        attachments_mixin.jira._session.get.assert_not_called()


class TestUploadPathTraversalRegression:
    """Regression — upload-side attachment path traversal / arbitrary file read.

    Covers GHSA-wm45, vc25, 93xw, 6cr4, f4p7, f6pj, 2xj6, mrq8, wv8v, p6hp, h7wj,
    mfv2, f26r, 9547, cc5h (read half), and the 6vmq download-overwrite variant.

    The download side gained ``validate_safe_path`` (CVE-2026-27825), but the
    upload path used to feed any caller-supplied ``file_path`` to the API after
    only an ``os.path.exists`` check, so a path outside the workspace was read and
    exfiltrated. These tests assert the secure outcome: a traversal/absolute path
    never reaches the upload sink.

    Assertions target the *sink* (``add_attachment`` not called), not an exception
    type, because ``upload_attachment`` wraps its body in ``except Exception`` and
    returns an error dict — so the fix may reject by raising or by returning a dict,
    and either way the secret must never reach the API.
    """

    @pytest.fixture
    def attachments_mixin(self, jira_fetcher: JiraFetcher) -> AttachmentsMixin:
        """AttachmentsMixin with a mocked Jira client (Cloud by default)."""
        mixin = jira_fetcher
        mixin.jira = MagicMock()
        mixin.jira._session = MagicMock()
        return mixin

    @pytest.mark.security_regression
    @pytest.mark.parametrize("attack", ["absolute_outside_cwd", "relative_traversal"])
    def test_upload_attachment_does_not_read_outside_workspace(
        self,
        attachments_mixin: AttachmentsMixin,
        tmp_path: Path,
        attack: str,
    ) -> None:
        """A file_path resolving outside the workspace must not reach the API."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        secret = tmp_path / "secret.txt"  # sibling of workspace -> outside it
        secret.write_bytes(b"SECRET-EXFIL")
        malicious = str(secret) if attack == "absolute_outside_cwd" else "../secret.txt"

        with patch("os.getcwd", return_value=str(workspace)):
            attachments_mixin.upload_attachment("PROJ-1", malicious)

        attachments_mixin.jira.add_attachment.assert_not_called()

    @pytest.mark.security_regression
    def test_upload_attachments_list_does_not_read_outside_workspace(
        self,
        attachments_mixin: AttachmentsMixin,
        tmp_path: Path,
    ) -> None:
        """The list/confused-deputy path (update_issue -> upload_attachments)."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_bytes(b"SECRET-EXFIL")

        with patch("os.getcwd", return_value=str(workspace)):
            attachments_mixin.upload_attachments("PROJ-1", [str(secret)])

        attachments_mixin.jira.add_attachment.assert_not_called()

    @pytest.mark.security_regression
    def test_download_into_cwd_module_is_rejected(
        self,
        attachments_mixin: AttachmentsMixin,
        tmp_path: Path,
    ) -> None:
        """Downloading onto a source file inside CWD must be confined/rejected."""
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"import os; os.system('id')\n"]
        mock_response.raise_for_status = MagicMock()
        attachments_mixin.jira._session.get.return_value = mock_response

        with (
            patch("builtins.open", mock_open()),
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=27),
            patch("os.makedirs"),
            patch("os.getcwd", return_value=str(tmp_path)),
        ):
            result = attachments_mixin.download_attachment(
                "https://test.url/evil",
                "important_module.py",  # relative -> resolves inside CWD
            )

        assert result is False, (
            "download targeting the CWD (where Python imports modules) must be "
            "confined to a dedicated directory, not allowed to overwrite source files"
        )


def _response(
    status: int = 200,
    body: bytes = b"",
    content_type: str = "application/octet-stream",
    url: str = "https://jira.example.com/x",
    history: list | None = None,
) -> MagicMock:
    """A requests.Response stand-in for the attachment transport."""
    response = MagicMock()
    response.status_code = status
    response.reason = {200: "OK", 502: "Bad Gateway", 404: "Not Found"}.get(
        status, "Error"
    )
    response.headers = {"Content-Type": content_type}
    response.url = url
    response.history = history or []
    response.iter_content.side_effect = lambda chunk_size=8192: iter([body])
    if status >= 400:
        response.raise_for_status.side_effect = Exception(f"{status} error")
    return response


class TestAttachmentDownloadFallbacks:
    """Regression tests for downloads refused in front of Jira Server/DC.

    An EC "Web Filter" answers 502 for every ``/secure/attachment/.../*.log``
    URL while the same bytes are served under a filter-safe spelling of the
    URL and inside the issue's "Download all" archive.
    """

    LOG_URL = (
        "https://jira.example.com/secure/attachment/356717/"
        "%5BSMT+IM1017179%5D+JB_DMA_510.log"
    )
    ENCODED_URL = (
        "https://jira.example.com/secure/attachment/356717/"
        "%5BSMT+IM1017179%5D+JB_DMA_510%2Elog"
    )
    BARE_URL = "https://jira.example.com/secure/attachment/356717/"

    @pytest.fixture
    def dc_mixin(self, jira_fetcher: JiraFetcher) -> AttachmentsMixin:
        mixin = jira_fetcher
        mixin.jira = MagicMock()
        mixin.jira._session = MagicMock()
        mixin.config = MagicMock(is_cloud=False, url="https://jira.example.com/")
        return mixin

    def test_url_candidates_for_server_dc(self):
        assert attachment_url_candidates(self.LOG_URL) == [
            self.LOG_URL,
            self.ENCODED_URL,
            self.BARE_URL,
        ]

    def test_url_candidates_keep_query_and_skip_cloud(self):
        url = "https://jira.example.com/secure/attachment/1/a.b.log?x=1"
        assert attachment_url_candidates(url) == [
            url,
            "https://jira.example.com/secure/attachment/1/a.b%2Elog?x=1",
            "https://jira.example.com/secure/attachment/1/?x=1",
        ]
        cloud = "https://x.atlassian.net/rest/api/3/attachment/content/10"
        assert attachment_url_candidates(cloud) == [cloud]

    def test_filter_block_is_retried_with_encoded_extension(self, dc_mixin):
        blocked = _response(
            502,
            b"<HTML><HEAD><TITLE>Web Filter</TITLE></HEAD>",
            content_type="text/html; charset=utf-8",
        )
        served = _response(200, b"log lines", content_type="text/plain")
        dc_mixin.jira._session.get.side_effect = [blocked, served]

        assert dc_mixin.fetch_attachment_content(self.LOG_URL) == b"log lines"

        urls = [c.args[0] for c in dc_mixin.jira._session.get.call_args_list]
        assert urls == [self.LOG_URL, self.ENCODED_URL]

    def test_failure_reports_status_and_page_title(self, dc_mixin):
        dc_mixin.jira._session.get.side_effect = lambda url, **kw: _response(
            502,
            b"<html><head><title>Web Filter</title></head></html>",
            content_type="text/html",
            url=url,
        )

        with pytest.raises(AttachmentFetchError) as excinfo:
            dc_mixin._open_attachment_with_fallback(self.LOG_URL)

        message = str(excinfo.value)
        assert 'HTTP 502 Bad Gateway (page title: "Web Filter")' in message
        assert message.count("; then ") == 2  # all three spellings were tried

    def test_transport_error_is_not_retried(self, dc_mixin):
        dc_mixin.jira._session.get.side_effect = ConnectionError("boom")

        with pytest.raises(AttachmentFetchError, match="boom"):
            dc_mixin._open_attachment_with_fallback(self.LOG_URL)

        assert dc_mixin.jira._session.get.call_count == 1

    def test_login_redirect_is_an_authentication_error(self, dc_mixin):
        login = _response(
            200,
            b"<html>login</html>",
            content_type="text/html",
            url="https://jira.example.com/login.jsp?os_destination=x",
            history=[_response(302)],
        )
        dc_mixin.jira._session.get.return_value = login

        with pytest.raises(AttachmentFetchError, match="login page"):
            dc_mixin._open_attachment(self.LOG_URL)

    @staticmethod
    def _archive_bytes(members: dict[str, bytes]) -> bytes:
        import io
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        return buffer.getvalue()

    def _issue_with_log(self, dc_mixin, size: int) -> None:
        dc_mixin.jira.issue.return_value = {
            "id": "306845",
            "fields": {
                "attachment": [
                    {
                        "id": "356717",
                        "filename": "svc.log",
                        "content": self.LOG_URL,
                        "size": size,
                        "mimeType": "text/plain",
                    },
                    {
                        "id": "1",
                        "filename": "shot.png",
                        "content": "https://jira.example.com/secure/attachment/1/shot.png",
                        "size": 3,
                        "mimeType": "image/png",
                    },
                ]
            },
        }

    def test_archive_fallback_serves_the_blocked_attachment(self, dc_mixin):
        """Every spelling is refused: the Download-all zip is used instead."""
        zip_bytes = self._archive_bytes({"svc.log": b"from zip", "shot.png": b"png"})

        def fake_get(url, **kwargs):
            if url.endswith("/secure/attachmentzip/306845.zip"):
                return _response(200, zip_bytes, content_type="application/zip")
            if "/secure/attachment/1/" in url:
                return _response(200, b"png", content_type="image/png")
            return _response(
                502, b"<title>Web Filter</title>", content_type="text/html"
            )

        dc_mixin.jira._session.get.side_effect = fake_get
        self._issue_with_log(dc_mixin, size=8)

        result = dc_mixin.get_issue_attachment_contents("ISSUE-1")

        assert result["failed"] == []
        by_name = {a["filename"]: a["data"] for a in result["attachments"]}
        assert by_name == {"svc.log": b"from zip", "shot.png": b"png"}
        zip_calls = [
            c.args[0]
            for c in dc_mixin.jira._session.get.call_args_list
            if "attachmentzip" in c.args[0]
        ]
        assert zip_calls == ["https://jira.example.com/secure/attachmentzip/306845.zip"]

    def test_archive_fallback_failure_is_reported_with_both_reasons(self, dc_mixin):
        dc_mixin.jira._session.get.side_effect = lambda url, **kw: _response(
            502, b"", url=url
        )
        self._issue_with_log(dc_mixin, size=8)

        result = dc_mixin.get_issue_attachment_contents("ISSUE-1", "*.log")

        assert result["skipped"] == ["shot.png"]
        assert len(result["failed"]) == 1
        error = result["failed"][0]["error"]
        assert "HTTP 502" in error
        assert "archive fallback failed" in error
        assert "attachmentzip/306845.zip" in error

    def test_archive_fallback_is_not_used_on_cloud(self, dc_mixin):
        dc_mixin.config.is_cloud = True
        dc_mixin.jira._session.get.side_effect = lambda url, **kw: _response(
            502, b"", url=url
        )
        self._issue_with_log(dc_mixin, size=8)

        result = dc_mixin.get_issue_attachment_contents("ISSUE-1", "*.log")

        assert "archive" not in result["failed"][0]["error"]
        assert not any(
            "attachmentzip" in c.args[0]
            for c in dc_mixin.jira._session.get.call_args_list
        )

    def test_download_to_disk_uses_archive_and_verifies_size(
        self, dc_mixin, tmp_path: Path
    ):
        zip_bytes = self._archive_bytes({"svc.log": b"from zip", "shot.png": b"png"})

        def fake_get(url, **kwargs):
            if url.endswith(".zip"):
                return _response(200, zip_bytes, content_type="application/zip")
            if "/secure/attachment/1/" in url:
                return _response(200, b"png", content_type="image/png")
            return _response(502, b"", url=url)

        dc_mixin.jira._session.get.side_effect = fake_get
        self._issue_with_log(dc_mixin, size=8)

        result = dc_mixin.download_issue_attachments(
            "ISSUE-1", "evidence/ISSUE-1", base_dir=tmp_path
        )

        assert result["failed"] == []
        assert (tmp_path / "evidence/ISSUE-1/svc.log").read_bytes() == b"from zip"
        assert (tmp_path / "evidence/ISSUE-1/shot.png").read_bytes() == b"png"
        assert {d["filename"]: d["size"] for d in result["downloaded"]} == {
            "svc.log": 8,
            "shot.png": 3,
        }

    def test_download_to_disk_discards_truncated_files(self, dc_mixin, tmp_path: Path):
        dc_mixin.jira._session.get.return_value = _response(
            200, b"short", content_type="text/plain"
        )
        dc_mixin.jira.issue.return_value = {
            "id": "1",
            "fields": {
                "attachment": [
                    {
                        "filename": "big.txt",
                        "content": "https://jira.example.com/secure/attachment/2/big.txt",
                        "size": 1000,
                    }
                ]
            },
        }

        result = dc_mixin.download_issue_attachments(
            "ISSUE-1", "out", base_dir=tmp_path
        )

        assert result["downloaded"] == []
        assert "Jira reports 1000 bytes" in result["failed"][0]["error"]
        assert not (tmp_path / "out/big.txt").exists()

    @pytest.mark.parametrize("target_dir", ["../outside", "/etc", "."])
    def test_download_to_disk_is_confined_to_base_dir(
        self, dc_mixin, tmp_path: Path, target_dir: str
    ):
        with pytest.raises(ValueError):
            dc_mixin.download_issue_attachments(
                "ISSUE-1", target_dir, base_dir=tmp_path
            )
        dc_mixin.jira.issue.assert_not_called()

    def test_filename_pattern_is_case_insensitive(self, dc_mixin, tmp_path: Path):
        dc_mixin.jira._session.get.return_value = _response(200, b"abc")
        dc_mixin.jira.issue.return_value = {
            "id": "1",
            "fields": {
                "attachment": [
                    {
                        "filename": "Report.PDF",
                        "content": "https://jira.example.com/secure/attachment/3/Report.PDF",
                        "size": 3,
                    },
                    {
                        "filename": "notes.txt",
                        "content": "https://jira.example.com/secure/attachment/4/notes.txt",
                        "size": 3,
                    },
                ]
            },
        }

        result = dc_mixin.download_issue_attachments(
            "ISSUE-1", "out", filename_pattern="*.pdf", base_dir=tmp_path
        )

        assert [d["filename"] for d in result["downloaded"]] == ["Report.PDF"]
        assert result["skipped"] == ["notes.txt"]
        assert result["total"] == 2
