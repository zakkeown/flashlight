"""
Tests for flashlight.utils.collect_env module.

Tests environment information collection.
"""

import unittest
from unittest.mock import patch

from flashlight.utils.collect_env import (
    SystemEnv,
    get_cpu_info,
    get_env_info,
    get_flashlight_version,
    get_gpu_info,
    get_mac_version,
    get_metal_available,
    get_mlx_version,
    get_numpy_version,
    get_os,
    get_os_version,
    get_pretty_env_info,
    get_python_platform,
    get_python_version,
)


class TestPythonInfo(unittest.TestCase):
    """Test Python information functions."""

    def test_get_python_version(self):
        """Test get_python_version returns valid version string."""
        version = get_python_version()
        self.assertIsInstance(version, str)
        parts = version.split(".")
        self.assertEqual(len(parts), 3)
        # All parts should be numeric
        for part in parts:
            self.assertTrue(part.isdigit())

    def test_get_python_platform(self):
        """Test get_python_platform returns valid platform."""
        platform = get_python_platform()
        self.assertIsInstance(platform, str)
        self.assertIn(platform, ["darwin", "linux", "win32", "cygwin"])


class TestOSInfo(unittest.TestCase):
    """Test OS information functions."""

    def test_get_os(self):
        """Test get_os returns valid OS name."""
        os_name = get_os()
        self.assertIsInstance(os_name, str)
        self.assertIn(os_name, ["Darwin", "Linux", "Windows"])

    def test_get_os_version(self):
        """Test get_os_version returns a string."""
        version = get_os_version()
        self.assertIsInstance(version, str)
        self.assertGreater(len(version), 0)

    def test_get_mac_version(self):
        """Test get_mac_version returns macOS info."""
        version = get_mac_version()
        self.assertIsInstance(version, str)
        self.assertIn("macOS", version)


class TestHardwareInfo(unittest.TestCase):
    """Test hardware information functions."""

    def test_get_cpu_info(self):
        """Test get_cpu_info returns CPU info."""
        cpu = get_cpu_info()
        self.assertIsInstance(cpu, str)
        # Should return something, even if just "Unknown"
        self.assertGreater(len(cpu), 0)

    def test_get_gpu_info(self):
        """Test get_gpu_info returns GPU info."""
        gpu = get_gpu_info()
        self.assertIsInstance(gpu, str)

    def test_get_metal_available(self):
        """Test get_metal_available returns boolean."""
        available = get_metal_available()
        self.assertIsInstance(available, bool)


class TestLibraryVersions(unittest.TestCase):
    """Test library version functions."""

    def test_get_mlx_version(self):
        """Test get_mlx_version returns version string."""
        version = get_mlx_version()
        self.assertIsInstance(version, str)
        # Should be a version or "Not installed"
        self.assertGreater(len(version), 0)

    def test_get_flashlight_version(self):
        """Test get_flashlight_version returns version string."""
        version = get_flashlight_version()
        self.assertIsInstance(version, str)

    def test_get_numpy_version(self):
        """Test get_numpy_version returns version string."""
        version = get_numpy_version()
        self.assertIsInstance(version, str)
        # NumPy should be installed
        self.assertNotEqual(version, "Not installed")


class TestGetEnvInfo(unittest.TestCase):
    """Test get_env_info function."""

    def test_returns_system_env(self):
        """Test get_env_info returns SystemEnv namedtuple."""
        env = get_env_info()
        self.assertIsInstance(env, SystemEnv)

    def test_all_fields_present(self):
        """Test all fields are present in env info."""
        env = get_env_info()

        self.assertIsNotNone(env.flashlight_version)
        self.assertIsNotNone(env.mlx_version)
        self.assertIsNotNone(env.python_version)
        self.assertIsNotNone(env.python_platform)
        self.assertIsNotNone(env.os)
        self.assertIsNotNone(env.os_version)
        self.assertIsNotNone(env.cpu_info)
        self.assertIsNotNone(env.gpu_info)
        self.assertIsInstance(env.metal_available, bool)
        self.assertIsNotNone(env.numpy_version)

    def test_python_version_matches(self):
        """Test Python version in env matches get_python_version."""
        env = get_env_info()
        self.assertEqual(env.python_version, get_python_version())


class TestGetPrettyEnvInfo(unittest.TestCase):
    """Test get_pretty_env_info function."""

    def test_returns_string(self):
        """Test get_pretty_env_info returns a string."""
        info = get_pretty_env_info()
        self.assertIsInstance(info, str)

    def test_contains_header(self):
        """Test output contains header."""
        info = get_pretty_env_info()
        self.assertIn("MLX Compat Environment", info)

    def test_contains_versions(self):
        """Test output contains version information."""
        info = get_pretty_env_info()
        self.assertIn("flashlight version:", info)
        self.assertIn("MLX version:", info)
        self.assertIn("Python version:", info)
        self.assertIn("NumPy version:", info)

    def test_contains_hardware_info(self):
        """Test output contains hardware information."""
        info = get_pretty_env_info()
        self.assertIn("CPU:", info)
        self.assertIn("GPU:", info)
        self.assertIn("Metal available:", info)

    def test_contains_os_info(self):
        """Test output contains OS information."""
        info = get_pretty_env_info()
        self.assertIn("OS:", info)
        self.assertIn("OS version:", info)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in collection functions."""

    def test_cpu_info_returns_string(self):
        """Test get_cpu_info always returns a string."""
        cpu = get_cpu_info()
        # Should return something, not raise
        self.assertIsInstance(cpu, str)
        self.assertGreater(len(cpu), 0)

    def test_gpu_info_returns_string(self):
        """Test get_gpu_info always returns a string."""
        gpu = get_gpu_info()
        # Should return a string (even if "Unknown")
        self.assertIsInstance(gpu, str)

    def test_metal_available_returns_bool(self):
        """Test get_metal_available returns a boolean."""
        available = get_metal_available()
        self.assertIsInstance(available, bool)


if __name__ == "__main__":
    unittest.main()
