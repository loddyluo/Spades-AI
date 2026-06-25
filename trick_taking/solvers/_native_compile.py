"""Shared compile helpers for native solver extensions."""

from __future__ import annotations

import subprocess
import sys


def compile_fastest_solver(src: str, out: str) -> None:
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

    flags = [
        compiler,
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
        # MinGW-w64 g++: -fPIC is a no-op (warns), and the default dynamic link
        # against libstdc++-6.dll / libgcc_s_seh-1.dll / libwinpthread-1.dll
        # makes ctypes.CDLL fail unless those DLLs are on PATH. Statically embed
        # the runtime so the produced .so is self-contained. -flto is dropped to
        # avoid MinGW's flaky linker-plugin path.
        flags = [
            compiler,
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
    flags.extend([src, "-o", out])
    subprocess.check_call(flags)


def compile_native_solver(src: str, out: str) -> None:
    subprocess.check_call([
        "g++",
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        src,
        "-o",
        out,
    ])


def compile_opt1_solver(src: str, out: str) -> None:
    subprocess.check_call([
        "g++",
        "-O3",
        "-march=native",
        "-flto",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-pthread",
        src,
        "-o",
        out,
    ])
