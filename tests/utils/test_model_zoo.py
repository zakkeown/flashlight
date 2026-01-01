"""
Tests for flashlight.utils.model_zoo module.

Tests model weight downloading and caching functionality.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from flashlight.utils.model_zoo import (
    get_dir,
    set_dir,
    load_url,
    download_url_to_file,
    _download_url_to_file,
)


class TestGetSetDir(unittest.TestCase):
    """Test get_dir and set_dir functions."""

    def test_get_dir_default(self):
        """Test get_dir returns a valid path."""
        # Clear env vars for test
        with patch.dict(os.environ, {}, clear=True):
            d = get_dir()
            self.assertIsInstance(d, str)
            self.assertIn("hub", d)

    def test_get_dir_with_mlx_home(self):
        """Test get_dir uses MLX_HOME env var."""
        with patch.dict(os.environ, {"MLX_HOME": "/tmp/mlx_test"}):
            d = get_dir()
            self.assertEqual(d, "/tmp/mlx_test/hub")

    def test_get_dir_with_xdg_cache(self):
        """Test get_dir uses XDG_CACHE_HOME env var."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": "/tmp/xdg_test"}, clear=True):
            # Remove MLX_HOME to ensure XDG is used
            if "MLX_HOME" in os.environ:
                del os.environ["MLX_HOME"]
            d = get_dir()
            self.assertIn("xdg_test", d)

    def test_set_dir(self):
        """Test set_dir changes the default directory."""
        original = get_dir()
        try:
            set_dir("/tmp/new_cache")
            # After set_dir, get_dir should return the new path
            # (unless env vars override)
            with patch.dict(os.environ, {}, clear=True):
                d = get_dir()
                self.assertEqual(d, "/tmp/new_cache")
        finally:
            # Restore original
            set_dir(original)


class TestDownloadUrlToFile(unittest.TestCase):
    """Test download functionality."""

    def test_download_creates_directory(self):
        """Test that download creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = os.path.join(tmpdir, "subdir", "file.txt")

            # Mock urlopen to return test content
            mock_response = MagicMock()
            mock_response.info.return_value = MagicMock()
            mock_response.info.return_value.get_all.return_value = ["100"]
            mock_response.read.side_effect = [b"test content", b""]

            with patch("flashlight.utils.model_zoo.urlopen", return_value=mock_response):
                _download_url_to_file("http://example.com/file.txt", dst, progress=False)

            self.assertTrue(os.path.exists(dst))
            with open(dst) as f:
                self.assertEqual(f.read(), "test content")

    def test_download_with_hash_verification(self):
        """Test hash verification during download."""
        import hashlib

        content = b"test content for hash"
        expected_hash = hashlib.sha256(content).hexdigest()[:8]

        with tempfile.TemporaryDirectory() as tmpdir:
            dst = os.path.join(tmpdir, "file.txt")

            mock_response = MagicMock()
            mock_response.info.return_value = MagicMock()
            mock_response.info.return_value.get_all.return_value = None
            mock_response.read.side_effect = [content, b""]

            with patch("flashlight.utils.model_zoo.urlopen", return_value=mock_response):
                # Should not raise with correct hash
                _download_url_to_file(
                    "http://example.com/file.txt",
                    dst,
                    hash_prefix=expected_hash,
                    progress=False,
                )

    def test_download_hash_mismatch(self):
        """Test that hash mismatch raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = os.path.join(tmpdir, "file.txt")

            mock_response = MagicMock()
            mock_response.info.return_value = MagicMock()
            mock_response.info.return_value.get_all.return_value = None
            mock_response.read.side_effect = [b"test content", b""]

            with patch("flashlight.utils.model_zoo.urlopen", return_value=mock_response):
                with self.assertRaises(RuntimeError) as ctx:
                    _download_url_to_file(
                        "http://example.com/file.txt",
                        dst,
                        hash_prefix="wronghash",
                        progress=False,
                    )
                self.assertIn("Hash mismatch", str(ctx.exception))


class TestLoadUrl(unittest.TestCase):
    """Test load_url function."""

    def test_load_url_uses_cache(self):
        """Test that load_url caches downloaded files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake cached file
            cached_file = os.path.join(tmpdir, "model.safetensors")

            # Create a minimal safetensors file
            import struct

            # Safetensors format: 8 bytes header size + header + data
            header = b'{"test": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}'
            header_size = len(header)
            data = struct.pack("<ff", 1.0, 2.0)

            with open(cached_file, "wb") as f:
                f.write(struct.pack("<Q", header_size))
                f.write(header)
                f.write(data)

            # Mock to return our temp directory
            with patch("flashlight.utils.model_zoo.get_dir", return_value=tmpdir):
                with patch("flashlight.utils.model_zoo._download_url_to_file") as mock_download:
                    # Should not download since file exists
                    try:
                        result = load_url("http://example.com/model.safetensors")
                    except Exception:
                        # May fail to load, but download should not be called
                        pass
                    mock_download.assert_not_called()

    def test_load_url_custom_filename(self):
        """Test load_url with custom filename creates correct cache path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the custom file so it exists (to avoid actual download)
            custom_file = os.path.join(tmpdir, "custom_name.bin")

            # Create a valid safetensors-like file
            import struct
            header = b'{"test": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}'
            header_size = len(header)
            data = struct.pack("<ff", 1.0, 2.0)

            with open(custom_file, "wb") as f:
                f.write(struct.pack("<Q", header_size))
                f.write(header)
                f.write(data)

            with patch("flashlight.utils.model_zoo.get_dir", return_value=tmpdir):
                with patch("flashlight.utils.model_zoo._download_url_to_file") as mock_download:
                    # File exists, so download should not be called
                    try:
                        result = load_url(
                            "http://example.com/model.bin",
                            file_name="custom_name.bin",
                        )
                    except Exception:
                        pass  # May fail to load, that's ok

                    # Download should not have been called since file exists
                    mock_download.assert_not_called()


class TestDownloadUrlToFilePublic(unittest.TestCase):
    """Test public download_url_to_file function."""

    def test_download_url_to_file_wrapper(self):
        """Test that public function is a wrapper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = os.path.join(tmpdir, "file.txt")

            mock_response = MagicMock()
            mock_response.info.return_value = MagicMock()
            mock_response.info.return_value.get_all.return_value = None
            mock_response.read.side_effect = [b"content", b""]

            with patch("flashlight.utils.model_zoo.urlopen", return_value=mock_response):
                download_url_to_file("http://example.com/file.txt", dst, progress=False)

            self.assertTrue(os.path.exists(dst))


if __name__ == "__main__":
    unittest.main()
