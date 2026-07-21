"""Resolve verified, architecture-specific native solver extensions."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from typing import Optional


NATIVE_LIBRARY_ABI_VERSION = 1
_IDENTITY_SYMBOL = "spades_native_build_id"
_ABI_SYMBOL = "spades_native_abi_version"


class NativeLibraryError(RuntimeError):
    """Raised when no binary matching the current native contract is available."""


def compute_native_build_id(
    src: str,
    *,
    required_symbols: tuple[str, ...],
    abi_version: int,
    build_recipe: str,
    target_platform: Optional[str] = None,
) -> str:
    """Hash every input that determines native-library compatibility."""
    metadata = {
        "schema_version": 1,
        "abi_version": int(abi_version),
        "build_recipe": build_recipe,
        "platform": target_platform or platform_tag(),
        "required_symbols": sorted(set(required_symbols)),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\0")
    with open(src, "rb") as source_file:
        digest.update(source_file.read())
    return digest.hexdigest()


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


def versioned_library_path(directory: str, base_name: str, build_id: str) -> str:
    """Return the ignored, content-addressed path loaded by the main process."""
    return os.path.join(
        directory,
        "__pycache__",
        "native",
        f"{base_name}.{platform_tag()}.{build_id}.so",
    )


def legacy_library_path(directory: str, base_name: str) -> str:
    return os.path.join(directory, f"{base_name}.so")


_PROBE_SCRIPT = r"""
import ctypes
import json
import sys

request = json.loads(sys.argv[1])
try:
    library = ctypes.CDLL(request["path"])
    for symbol in request["required_symbols"]:
        getattr(library, symbol)

    build_id_function = getattr(library, "spades_native_build_id")
    build_id_function.argtypes = []
    build_id_function.restype = ctypes.c_char_p
    raw_build_id = build_id_function()
    actual_build_id = raw_build_id.decode("ascii") if raw_build_id else ""

    abi_function = getattr(library, "spades_native_abi_version")
    abi_function.argtypes = []
    abi_function.restype = ctypes.c_uint32
    actual_abi_version = int(abi_function())

    if actual_build_id != request["expected_build_id"]:
        raise RuntimeError(
            "build ID mismatch: "
            + actual_build_id
            + " != "
            + request["expected_build_id"]
        )
    if actual_abi_version != request["expected_abi_version"]:
        raise RuntimeError(
            "ABI version mismatch: "
            + str(actual_abi_version)
            + " != "
            + str(request["expected_abi_version"])
        )
except BaseException as error:
    print(str(error), file=sys.stderr)
    raise SystemExit(1)
"""


def _probe_native_library(
    path: str,
    *,
    expected_build_id: str,
    expected_abi_version: int,
    required_symbols: tuple[str, ...],
) -> tuple[bool, str]:
    """Validate a candidate in a child process so the parent never loads it."""
    if not os.path.isfile(path):
        return False, "file does not exist"

    request = json.dumps(
        {
            "path": os.path.abspath(path),
            "expected_build_id": expected_build_id,
            "expected_abi_version": int(expected_abi_version),
            "required_symbols": list(required_symbols),
        },
        separators=(",", ":"),
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT, request],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"probe failed: {error}"

    if completed.returncode == 0:
        return True, "ok"
    reason = completed.stderr.strip()
    if not reason:
        reason = f"probe process exited with status {completed.returncode}"
    return False, reason


def _atomic_copy(source: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=os.path.dirname(destination),
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ensure_native_library(
    directory: str,
    base_name: str,
    src_filename: str,
    compile_fn: Callable[[str, str, str, int], None],
    *,
    required_symbols: tuple[str, ...],
    abi_version: int,
    build_recipe: str,
) -> str:
    """Find or build a native library matching the exact current contract."""
    src = os.path.join(directory, src_filename)
    if not os.path.isfile(src):
        raise NativeLibraryError(f"native solver source does not exist: {src}")

    build_id = compute_native_build_id(
        src,
        required_symbols=required_symbols,
        abi_version=abi_version,
        build_recipe=build_recipe,
    )
    versioned = versioned_library_path(directory, base_name, build_id)
    probe_arguments = {
        "expected_build_id": build_id,
        "expected_abi_version": abi_version,
        "required_symbols": required_symbols,
    }
    diagnostics: list[str] = []

    if os.path.isfile(versioned):
        valid, reason = _probe_native_library(versioned, **probe_arguments)
        if valid:
            return versioned
        diagnostics.append(f"{versioned}: {reason}")

    prebuilt_candidates = [
        arch_library_path(directory, base_name),
        legacy_library_path(directory, base_name),
    ]
    for candidate in prebuilt_candidates:
        if not os.path.isfile(candidate):
            continue
        valid, reason = _probe_native_library(candidate, **probe_arguments)
        if not valid:
            diagnostics.append(f"{candidate}: {reason}")
            continue

        _atomic_copy(candidate, versioned)
        copied_valid, copied_reason = _probe_native_library(
            versioned, **probe_arguments
        )
        if copied_valid:
            return versioned
        diagnostics.append(f"{versioned}: {copied_reason}")

    os.makedirs(os.path.dirname(versioned), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{base_name}.{platform_tag()}.",
        suffix=".so",
        dir=os.path.dirname(versioned),
    )
    os.close(descriptor)
    try:
        try:
            compile_fn(src, temporary, build_id, abi_version)
        except Exception as error:
            rejected = "; ".join(diagnostics) or "no candidate binary found"
            raise NativeLibraryError(
                f"failed to build verified native library {base_name} for "
                f"{platform_tag()}: {error}; rejected candidates: {rejected}"
            ) from error

        valid, reason = _probe_native_library(temporary, **probe_arguments)
        if not valid:
            raise NativeLibraryError(
                f"compiler produced an invalid native library for {base_name}: {reason}"
            )

        os.replace(temporary, versioned)
        valid, reason = _probe_native_library(versioned, **probe_arguments)
        if not valid:
            raise NativeLibraryError(
                f"promoted native library failed verification for {base_name}: {reason}"
            )
        return versioned
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
