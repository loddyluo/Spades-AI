"""Shared compile helpers for native solver extensions."""

from __future__ import annotations

import json
import subprocess
import sys


def _recipe(name: str, flags: list[str]) -> str:
    return json.dumps(
        {"name": name, "revision": 1, "flags": flags},
        sort_keys=True,
        separators=(",", ":"),
    )


def _fastest_flags() -> list[str]:
    flags = [
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-pthread",
        "-DNDEBUG",
        "-fomit-frame-pointer",
        "-funroll-loops",
    ]
    if sys.platform == "darwin":
        flags.append("-march=native")
    elif sys.platform == "win32":
        # MinGW-w64 g++: -fPIC is a no-op (warns), and dynamic runtime DLLs
        # are not reliably present beside the application.
        flags = [
            "-O3",
            "-std=c++17",
            "-shared",
            "-pthread",
            "-DNDEBUG",
            "-fomit-frame-pointer",
            "-funroll-loops",
            "-march=native",
            "-static",
            "-static-libgcc",
            "-static-libstdc++",
        ]
    else:
        flags.extend(["-march=native", "-flto"])
    return flags


def _native_flags() -> list[str]:
    return ["-O3", "-std=c++17", "-shared", "-fPIC"]


def _opt1_flags() -> list[str]:
    return [
        "-O3",
        "-march=native",
        "-flto",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-pthread",
    ]


FASTEST_BUILD_RECIPE = _recipe("fastest", _fastest_flags())
NATIVE_BUILD_RECIPE = _recipe("native", _native_flags())
OPT1_BUILD_RECIPE = _recipe("opt1", _opt1_flags())


def _identity_flags(build_id: str, abi_version: int) -> list[str]:
    return [
        f'-DSPADES_NATIVE_BUILD_ID="{build_id}"',
        f"-DSPADES_NATIVE_ABI_VERSION={int(abi_version)}",
    ]


def compile_fastest_solver(
    src: str,
    out: str,
    build_id: str,
    abi_version: int,
) -> None:
    compiler = "g++"
    try:
        subprocess.check_call(
            ["clang++", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        compiler = "clang++"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    command = [compiler, *_fastest_flags(), *_identity_flags(build_id, abi_version)]
    command.extend([src, "-o", out])
    subprocess.check_call(command)


def compile_native_solver(
    src: str,
    out: str,
    build_id: str,
    abi_version: int,
) -> None:
    command = ["g++", *_native_flags(), *_identity_flags(build_id, abi_version)]
    command.extend([src, "-o", out])
    subprocess.check_call(command)


def compile_opt1_solver(
    src: str,
    out: str,
    build_id: str,
    abi_version: int,
) -> None:
    command = ["g++", *_opt1_flags(), *_identity_flags(build_id, abi_version)]
    command.extend([src, "-o", out])
    subprocess.check_call(command)
