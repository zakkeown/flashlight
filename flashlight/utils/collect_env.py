"""
Environment collection utilities for debugging.

Provides system and library information for debugging and issue reporting.
"""

import os
import platform
import re
import subprocess
import sys
from collections import namedtuple
from typing import Optional, Tuple


# Named tuple for environment info
SystemEnv = namedtuple(
    "SystemEnv",
    [
        "flashlight_version",
        "mlx_version",
        "python_version",
        "python_platform",
        "os",
        "os_version",
        "cpu_info",
        "gpu_info",
        "metal_available",
        "numpy_version",
        "pip_packages",
        "conda_packages",
    ],
)


def get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_python_platform() -> str:
    """Get Python platform info."""
    return sys.platform


def get_os() -> str:
    """Get operating system name."""
    return platform.system()


def get_os_version() -> str:
    """Get operating system version."""
    if platform.system() == "Darwin":
        return get_mac_version()
    elif platform.system() == "Linux":
        return get_linux_version()
    elif platform.system() == "Windows":
        return platform.version()
    return platform.release()


def get_mac_version() -> str:
    """Get macOS version string."""
    try:
        version = platform.mac_ver()[0]
        return f"macOS {version}"
    except Exception:
        return "macOS (version unknown)"


def get_linux_version() -> str:
    """Get Linux distribution info."""
    try:
        # Try to read /etc/os-release
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=")[1].strip().strip('"')
    except Exception:
        pass

    try:
        return platform.linux_distribution()[0]
    except Exception:
        return platform.release()


def get_cpu_info() -> str:
    """Get CPU information."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        # Try to get Apple Silicon info
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.chip"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass

    return platform.processor() or "Unknown"


def get_gpu_info() -> str:
    """Get GPU information (Metal device for macOS)."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                # Parse GPU name from output
                for line in result.stdout.split("\n"):
                    if "Chipset Model:" in line:
                        return line.split(":")[1].strip()
                    if "Chip:" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass

    return "Unknown"


def get_metal_available() -> bool:
    """Check if Metal is available."""
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except Exception:
        return False


def get_mlx_version() -> str:
    """Get MLX version."""
    try:
        import mlx
        return mlx.__version__
    except Exception:
        return "Not installed"


def get_flashlight_version() -> str:
    """Get flashlight version."""
    try:
        import flashlight
        return getattr(flashlight, "__version__", "Unknown")
    except Exception:
        return "Not installed"


def get_numpy_version() -> str:
    """Get NumPy version."""
    try:
        import numpy
        return numpy.__version__
    except Exception:
        return "Not installed"


def run_and_return_output(cmd: list) -> str:
    """Run a command and return its output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def get_pip_packages() -> str:
    """Get relevant pip packages."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # Filter for relevant packages
            relevant = ["mlx", "numpy", "scipy", "torch", "tensorflow", "jax"]
            lines = result.stdout.strip().split("\n")
            filtered = []
            for line in lines:
                for pkg in relevant:
                    if line.lower().startswith(pkg):
                        filtered.append(line)
                        break
            return "\n".join(filtered)
    except Exception:
        pass
    return ""


def get_conda_packages() -> str:
    """Get conda packages if in conda environment."""
    if "CONDA_PREFIX" not in os.environ:
        return ""

    try:
        result = subprocess.run(
            ["conda", "list"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # Filter for relevant packages
            relevant = ["mlx", "numpy", "scipy", "python"]
            lines = result.stdout.strip().split("\n")
            filtered = []
            for line in lines:
                for pkg in relevant:
                    if line.lower().startswith(pkg):
                        filtered.append(line)
                        break
            return "\n".join(filtered)
    except Exception:
        pass
    return ""


def get_env_info() -> SystemEnv:
    """
    Collect environment information.

    Returns:
        SystemEnv namedtuple with all environment info.
    """
    return SystemEnv(
        flashlight_version=get_flashlight_version(),
        mlx_version=get_mlx_version(),
        python_version=get_python_version(),
        python_platform=get_python_platform(),
        os=get_os(),
        os_version=get_os_version(),
        cpu_info=get_cpu_info(),
        gpu_info=get_gpu_info(),
        metal_available=get_metal_available(),
        numpy_version=get_numpy_version(),
        pip_packages=get_pip_packages(),
        conda_packages=get_conda_packages(),
    )


def get_pretty_env_info() -> str:
    """
    Get a formatted string of environment information.

    Returns:
        Formatted multi-line string with environment details.
    """
    env = get_env_info()

    lines = [
        "MLX Compat Environment",
        "=" * 50,
        "",
        f"flashlight version: {env.flashlight_version}",
        f"MLX version: {env.mlx_version}",
        f"Metal available: {env.metal_available}",
        "",
        f"Python version: {env.python_version}",
        f"Python platform: {env.python_platform}",
        "",
        f"OS: {env.os}",
        f"OS version: {env.os_version}",
        "",
        f"CPU: {env.cpu_info}",
        f"GPU: {env.gpu_info}",
        "",
        f"NumPy version: {env.numpy_version}",
    ]

    if env.pip_packages:
        lines.extend(["", "Relevant pip packages:", env.pip_packages])

    if env.conda_packages:
        lines.extend(["", "Relevant conda packages:", env.conda_packages])

    return "\n".join(lines)


def main() -> None:
    """Print environment info to stdout."""
    print(get_pretty_env_info())


if __name__ == "__main__":
    main()
