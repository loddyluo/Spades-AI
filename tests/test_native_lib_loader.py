from __future__ import annotations

import os

import pytest

from trick_taking.solvers import native_lib_loader as loader
from trick_taking.solvers import _native_compile as native_compile
from trick_taking.solvers.native_lib_loader import compute_native_build_id


def test_build_id_changes_for_every_compatibility_input(tmp_path) -> None:
    source = tmp_path / "solver.cpp"
    source.write_text("int solver = 1;", encoding="utf-8")

    def build_id(
        *,
        symbols: tuple[str, ...] = ("solve_native",),
        abi_version: int = 1,
        recipe: str = "cxx17-o3-v1",
        target: str = "linux_x86_64",
    ) -> str:
        return compute_native_build_id(
            str(source),
            required_symbols=symbols,
            abi_version=abi_version,
            build_recipe=recipe,
            target_platform=target,
        )

    baseline = build_id()

    assert len(baseline) == 64
    assert baseline == build_id()
    assert baseline != build_id(symbols=("solve_native_with_q",))
    assert baseline != build_id(abi_version=2)
    assert baseline != build_id(recipe="cxx17-o3-v2")
    assert baseline != build_id(target="darwin_arm64")

    source.write_text("int solver = 2;", encoding="utf-8")
    assert baseline != build_id()


def _identity(tmp_path) -> tuple[str, tuple[str, ...], int, str, str]:
    source = tmp_path / "solver.cpp"
    source.write_text("extern \"C\" int solve_native() { return 1; }", encoding="utf-8")
    required_symbols = ("solve_native",)
    abi_version = 1
    recipe = "test-cxx17-v1"
    build_id = compute_native_build_id(
        str(source),
        required_symbols=required_symbols,
        abi_version=abi_version,
        build_recipe=recipe,
    )
    return str(source), required_symbols, abi_version, recipe, build_id


def test_valid_prebuilt_is_copied_to_content_addressed_load_path(
    tmp_path, monkeypatch
) -> None:
    source, symbols, abi_version, recipe, build_id = _identity(tmp_path)
    prebuilt = loader.arch_library_path(str(tmp_path), "_solver_core")
    with open(prebuilt, "wb") as binary:
        binary.write(b"verified-prebuilt")

    probed_paths: list[str] = []

    def probe(path: str, **expected) -> tuple[bool, str]:
        probed_paths.append(path)
        assert expected == {
            "expected_build_id": build_id,
            "expected_abi_version": abi_version,
            "required_symbols": symbols,
        }
        return True, "ok"

    def must_not_compile(*args) -> None:
        raise AssertionError("a valid prebuilt must not be recompiled")

    monkeypatch.setattr(loader, "_probe_native_library", probe, raising=False)

    result = loader.ensure_native_library(
        str(tmp_path),
        "_solver_core",
        os.path.basename(source),
        must_not_compile,
        required_symbols=symbols,
        abi_version=abi_version,
        build_recipe=recipe,
    )

    assert result == loader.versioned_library_path(
        str(tmp_path), "_solver_core", build_id
    )
    assert result != prebuilt
    with open(result, "rb") as binary:
        assert binary.read() == b"verified-prebuilt"
    assert probed_paths == [prebuilt, result]


def test_old_prebuilt_is_rejected_and_recompiled_with_current_identity(
    tmp_path, monkeypatch
) -> None:
    source, symbols, abi_version, recipe, build_id = _identity(tmp_path)
    prebuilt = loader.arch_library_path(str(tmp_path), "_solver_core")
    with open(prebuilt, "wb") as binary:
        binary.write(b"old-but-loadable")

    probed_paths: list[str] = []
    compile_calls: list[tuple[str, str, str, int]] = []

    def probe(path: str, **expected) -> tuple[bool, str]:
        probed_paths.append(path)
        assert expected["expected_build_id"] == build_id
        if path == prebuilt:
            return False, "missing spades_native_build_id"
        return os.path.isfile(path), "ok"

    def compile_current(
        src: str, out: str, actual_build_id: str, actual_abi_version: int
    ) -> None:
        compile_calls.append((src, out, actual_build_id, actual_abi_version))
        with open(out, "wb") as binary:
            binary.write(b"current")

    monkeypatch.setattr(loader, "_probe_native_library", probe, raising=False)

    result = loader.ensure_native_library(
        str(tmp_path),
        "_solver_core",
        os.path.basename(source),
        compile_current,
        required_symbols=symbols,
        abi_version=abi_version,
        build_recipe=recipe,
    )

    assert result == loader.versioned_library_path(
        str(tmp_path), "_solver_core", build_id
    )
    assert result != prebuilt
    assert len(compile_calls) == 1
    assert compile_calls[0][0] == source
    assert compile_calls[0][2:] == (build_id, abi_version)
    assert probed_paths[0] == prebuilt
    assert probed_paths[-1] == result


def test_compilation_failure_never_falls_back_to_old_prebuilt(
    tmp_path, monkeypatch
) -> None:
    source, symbols, abi_version, recipe, _build_id = _identity(tmp_path)
    prebuilt = loader.arch_library_path(str(tmp_path), "_solver_core")
    with open(prebuilt, "wb") as binary:
        binary.write(b"old-but-loadable")

    monkeypatch.setattr(
        loader,
        "_probe_native_library",
        lambda *args, **kwargs: (False, "build ID mismatch"),
        raising=False,
    )

    def compiler_missing(*args) -> None:
        raise FileNotFoundError("no C++ compiler")

    with pytest.raises(loader.NativeLibraryError, match="no C\\+\\+ compiler"):
        loader.ensure_native_library(
            str(tmp_path),
            "_solver_core",
            os.path.basename(source),
            compiler_missing,
            required_symbols=symbols,
            abi_version=abi_version,
            build_recipe=recipe,
        )


@pytest.mark.parametrize(
    "compile_function",
    [
        native_compile.compile_fastest_solver,
        native_compile.compile_native_solver,
        native_compile.compile_opt1_solver,
    ],
)
def test_compile_command_embeds_build_id_and_abi(
    compile_function, monkeypatch
) -> None:
    commands: list[list[str]] = []

    def record(command: list[str], **kwargs) -> None:
        commands.append(command)

    monkeypatch.setattr(native_compile.subprocess, "check_call", record)
    build_id = "a" * 64

    compile_function("solver.cpp", "solver.so", build_id, 1)

    command = commands[-1]
    assert f'-DSPADES_NATIVE_BUILD_ID="{build_id}"' in command
    assert "-DSPADES_NATIVE_ABI_VERSION=1" in command
    assert command[-3:] == ["solver.cpp", "-o", "solver.so"]
