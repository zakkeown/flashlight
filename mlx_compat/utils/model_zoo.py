"""
Model zoo utilities for downloading pretrained weights.

Provides PyTorch-compatible API for downloading model checkpoints.
"""

import hashlib
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse
from urllib.request import urlopen, Request
import shutil

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# Default cache directory
_DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mlx_compat", "hub")

# Environment variable for cache directory
ENV_MLX_HOME = "MLX_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"


def get_dir() -> str:
    """
    Get the directory where downloaded models are cached.

    Returns:
        str: Path to the cache directory.
    """
    # Check environment variables in order of precedence
    if ENV_MLX_HOME in os.environ:
        return os.path.join(os.environ[ENV_MLX_HOME], "hub")

    if ENV_XDG_CACHE_HOME in os.environ:
        return os.path.join(os.environ[ENV_XDG_CACHE_HOME], "mlx_compat", "hub")

    return _DEFAULT_CACHE_DIR


def set_dir(d: str) -> None:
    """
    Set the directory for caching downloaded models.

    Args:
        d: Path to the new cache directory.
    """
    global _DEFAULT_CACHE_DIR
    _DEFAULT_CACHE_DIR = d


def _download_url_to_file(
    url: str,
    dst: str,
    hash_prefix: Optional[str] = None,
    progress: bool = True,
) -> None:
    """
    Download a file from a URL to a destination path.

    Args:
        url: URL to download from.
        dst: Destination file path.
        hash_prefix: Expected hash prefix for verification.
        progress: Whether to show progress bar.
    """
    # Create request with user agent
    req = Request(url, headers={"User-Agent": "mlx_compat"})

    # Open URL
    u = urlopen(req)
    meta = u.info()

    # Get file size if available
    file_size = None
    if hasattr(meta, "get_all"):
        content_length = meta.get_all("Content-Length")
        if content_length is not None and len(content_length) > 0:
            file_size = int(content_length[0])

    # Create parent directory if needed
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # Download with optional progress bar
    sha256 = hashlib.sha256() if hash_prefix else None

    with open(dst, "wb") as f:
        if progress and HAS_TQDM and file_size:
            with tqdm(total=file_size, unit="B", unit_scale=True, desc=os.path.basename(dst)) as pbar:
                while True:
                    buffer = u.read(8192)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    if sha256:
                        sha256.update(buffer)
                    pbar.update(len(buffer))
        else:
            if progress and file_size:
                print(f"Downloading {os.path.basename(dst)} ({file_size / 1024 / 1024:.1f} MB)")
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if sha256:
                    sha256.update(buffer)

    # Verify hash if provided
    if hash_prefix and sha256:
        digest = sha256.hexdigest()
        if not digest.startswith(hash_prefix):
            raise RuntimeError(
                f"Hash mismatch for {url}. Expected {hash_prefix}, got {digest[:len(hash_prefix)]}"
            )


def load_url(
    url: str,
    model_dir: Optional[str] = None,
    map_location: Optional[str] = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None,
    weights_only: bool = False,
) -> Dict[str, Any]:
    """
    Load a model checkpoint from a URL.

    Downloads the checkpoint if not already cached, then loads and returns it.

    Args:
        url: URL of the checkpoint to download.
        model_dir: Directory to cache the downloaded file. Defaults to get_dir().
        map_location: Ignored (MLX uses unified memory).
        progress: Whether to show download progress.
        check_hash: Whether to verify the hash from the filename.
        file_name: Override the filename for caching.
        weights_only: If True, only load weights (safer for untrusted sources).

    Returns:
        Dict containing the loaded checkpoint.

    Example:
        >>> state_dict = load_url('https://example.com/model.safetensors')
        >>> model.load_state_dict(state_dict)
    """
    import mlx_compat

    # Determine cache directory
    if model_dir is None:
        model_dir = get_dir()

    os.makedirs(model_dir, exist_ok=True)

    # Determine filename
    if file_name is None:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)

    cached_file = os.path.join(model_dir, file_name)

    # Check hash from filename if requested
    hash_prefix = None
    if check_hash:
        # PyTorch convention: filename might contain hash like model-abc123.pth
        name_parts = file_name.split("-")
        if len(name_parts) > 1:
            # Extract potential hash from filename
            potential_hash = name_parts[-1].split(".")[0]
            if len(potential_hash) >= 8:
                hash_prefix = potential_hash

    # Download if not cached
    if not os.path.exists(cached_file):
        _download_url_to_file(url, cached_file, hash_prefix, progress)

    # Load the checkpoint
    return mlx_compat.load(cached_file, weights_only=weights_only)


def download_url_to_file(
    url: str,
    dst: str,
    hash_prefix: Optional[str] = None,
    progress: bool = True,
) -> None:
    """
    Download a file from a URL to a local path.

    This is a public wrapper around the internal download function.

    Args:
        url: URL to download from.
        dst: Destination file path.
        hash_prefix: Expected SHA256 hash prefix for verification.
        progress: Whether to show progress bar.
    """
    _download_url_to_file(url, dst, hash_prefix, progress)


# Aliases for compatibility
load = load_url
