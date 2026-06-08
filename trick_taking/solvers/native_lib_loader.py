"""Resolve and load architecture-specific native solver extensions."""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from collections.abc import Callable
from typing import Optional


def platform_tag() -> str:
    """Return a stable tag like ``darwin_arm64`` or ``linux_x86_64``."""
    system = sys.platform
    if system.startswith("linux"):
        system = "linux"
    elif system == "darwin":
        system = "darwin"
    elif system == "win32":
        system = "windows"

    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        machine = "x86_64"
    elif machine in ("arm64", "aarch64"):
        machine = "aarch64" if system == "linux" else "arm64"
    return f"{system}_{machine}"


def arch_library_path(directory: str, base_name: str) -> str:
    return os.path.join(directory, f"{base_name}.{platform_tag()}.so")


def legacy_library_path(directory: str, base_name: str) -> str:
    return os.path.join(directory, f"{base_name}.so")


def can_load_library(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return False
    # Drop the handle immediately; callers will reload with configured signatures.
    del lib
    return True


def is_stale(src: str, out: str) -> bool:
    if not os.path.isfile(out):
        return True
    if not os.path.isfile(src):
        return False
    return os.path.getmtime(src) > os.path.getmtime(out)


def ensure_native_library(
    directory: str,
    base_name: str,
    src_filename: str,
    compile_fn: Callable[[str, str], None],
) -> Optional[str]:
    """Find or build a loadable native library for the current platform."""
    src = os.path.join(directory, src_filename)
    candidates = [
        arch_library_path(directory, base_name),
        legacy_library_path(directory, base_name),
    ]

    for path in candidates:
        if os.path.isfile(path) and not is_stale(src, path) and can_load_library(path):
            return path

    out = arch_library_path(directory, base_name)
    try:
        compile_fn(src, out)
    except Exception:
        return None

    if can_load_library(out):
        return out
    return None
