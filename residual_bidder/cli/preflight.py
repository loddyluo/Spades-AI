"""Verify frozen inputs and runtime capabilities before formal bidder work."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from residual_bidder.config import BidderConfig, ConfigError, canonical_sha256


class PreflightError(RuntimeError):
    """Raised when a required artifact or runtime capability is unavailable."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise PreflightError(f"cannot read required file {path}: {error}") from error
    return digest.hexdigest()


def _verify_frozen_file(path_text: str, expected: str) -> str:
    path = Path(path_text)
    actual = _file_sha256(path)
    if actual != expected:
        raise PreflightError(
            f"{path.name} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _inspect_frozen_inputs(config: BidderConfig) -> dict[str, object]:
    nsfp_sha256 = _verify_frozen_file(config.nsfp.path, config.nsfp.sha256)
    play_config_sha256 = _verify_frozen_file(
        config.play.config_path, config.play.config_sha256
    )
    source_hashes: list[list[str]] = []
    for relative_path in config.play.source_manifest:
        source_hashes.append([relative_path, _file_sha256(Path(relative_path))])
    pipeline_payload: dict[str, object] = {
        "play_config_sha256": play_config_sha256,
        "source_manifest": source_hashes,
    }
    return {
        "config_sha256": config.sha256(),
        "nsfp_sha256": nsfp_sha256,
        "play_config_sha256": play_config_sha256,
        "play_source_hashes": source_hashes,
        "play_pipeline_sha256": canonical_sha256(pipeline_payload),
    }


def _load_torch(torch_module: Any | None) -> Any:
    if torch_module is not None:
        return torch_module
    try:
        import torch
    except ImportError as error:
        raise PreflightError("PyTorch is unavailable") from error
    return torch


def _finite(torch_module: Any, tensor: Any) -> bool:
    return bool(torch_module.isfinite(tensor).all().item())


def _run_cuda_probe(torch_module: Any) -> None:
    torch_module.manual_seed(20260721)
    left = torch_module.randn(
        32, 32, device="cuda", dtype=torch_module.float32, requires_grad=True
    )
    right = torch_module.randn(
        32, 32, device="cuda", dtype=torch_module.float32, requires_grad=True
    )
    output = torch_module.matmul(left, right)
    loss = output.square().mean()
    loss.backward()
    torch_module.cuda.synchronize()
    if not _finite(torch_module, output):
        raise PreflightError("CUDA probe produced a non-finite forward value")
    if left.grad is None or right.grad is None:
        raise PreflightError("CUDA probe did not produce gradients")
    if not _finite(torch_module, left.grad) or not _finite(torch_module, right.grad):
        raise PreflightError("CUDA probe produced a non-finite gradient")


def _inspect_cuda(
    torch_module: Any, *, require_cuda: bool, require_sm120: bool
) -> dict[str, object]:
    available = bool(torch_module.cuda.is_available())
    if (require_cuda or require_sm120) and not available:
        raise PreflightError("CUDA is required but unavailable")

    report: dict[str, object] = {
        "torch": str(getattr(torch_module, "__version__", "unknown")),
        "compiled_cuda": getattr(getattr(torch_module, "version", None), "cuda", None),
        "cuda_available": available,
        "gpu_name": None,
        "compute_capability": None,
        "architecture_list": [],
        "sm_120": False,
    }
    if not available:
        return report

    gpu_name = str(torch_module.cuda.get_device_name(0))
    capability = tuple(int(part) for part in torch_module.cuda.get_device_capability(0))
    architecture_list = list(torch_module.cuda.get_arch_list())
    has_sm120 = capability == (12, 0) and "sm_120" in architecture_list
    report.update(
        {
            "gpu_name": gpu_name,
            "compute_capability": list(capability),
            "architecture_list": architecture_list,
            "sm_120": has_sm120,
        }
    )
    if require_sm120 and "sm_120" not in architecture_list:
        raise PreflightError(
            f"PyTorch architecture list does not contain sm_120: {architecture_list}"
        )
    if require_sm120 and capability != (12, 0):
        raise PreflightError(
            f"GPU compute capability {capability} is not required capability (12, 0)"
        )
    _run_cuda_probe(torch_module)
    return report


def _default_solver_factory() -> Any:
    from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
        ExactDoubleDummyCppFastestSolver,
    )

    return ExactDoubleDummyCppFastestSolver()


def _native_build_id(solver: Any) -> str:
    explicit = getattr(solver, "native_build_id", None)
    if explicit is not None:
        return str(explicit)
    library = getattr(solver, "_lib", None)
    function = getattr(library, "spades_native_build_id", None)
    if function is None:
        raise PreflightError("loaded native solver does not expose its build ID")
    function.argtypes = []
    function.restype = ctypes.c_char_p
    raw = function()
    if not raw:
        raise PreflightError("loaded native solver returned an empty build ID")
    return raw.decode("ascii")


def _native_library_path(solver: Any) -> Path:
    explicit = getattr(solver, "native_library_path", None)
    if explicit is not None:
        return Path(explicit)
    library = getattr(solver, "_lib", None)
    loaded_path = getattr(library, "_name", None)
    if not loaded_path:
        raise PreflightError("loaded native solver does not expose its library path")
    return Path(loaded_path)


def _inspect_native_solver(
    solver_factory: Callable[[], Any] | None, *, required: bool
) -> dict[str, object]:
    report: dict[str, object] = {
        "native_solver": False,
        "native_solver_build_id": None,
        "native_library_path": None,
        "native_library_sha256": None,
    }
    if not required:
        return report
    factory = solver_factory or _default_solver_factory
    try:
        solver = factory()
    except Exception as error:
        raise PreflightError(f"failed to initialize native solver: {error}") from error
    if not bool(getattr(solver, "native_available", False)):
        raise PreflightError("native solver is required but unavailable")
    library_path = _native_library_path(solver)
    report.update(
        {
            "native_solver": True,
            "native_solver_build_id": _native_build_id(solver),
            "native_library_path": str(library_path),
            "native_library_sha256": _file_sha256(library_path),
        }
    )
    return report


def inspect_runtime(
    config: BidderConfig,
    require_cuda: bool,
    require_sm120: bool,
    *,
    torch_module: Any | None = None,
    solver_factory: Callable[[], Any] | None = None,
    require_native_solver: bool = False,
) -> dict[str, object]:
    """Inspect all frozen inputs and requested runtime capabilities."""
    frozen_report = _inspect_frozen_inputs(config)
    torch_runtime = _load_torch(torch_module)
    cuda_report = _inspect_cuda(
        torch_runtime, require_cuda=require_cuda, require_sm120=require_sm120
    )
    native_report = _inspect_native_solver(
        solver_factory, required=require_native_solver
    )
    return {
        "ok": True,
        "python": platform.python_version(),
        **cuda_report,
        **native_report,
        **frozen_report,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/residual_bidder/base.yaml"),
        help="strict bidder configuration to inspect",
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-sm120", action="store_true")
    parser.add_argument("--require-native-solver", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = BidderConfig.load(arguments.config)
        report = inspect_runtime(
            config,
            require_cuda=arguments.require_cuda,
            require_sm120=arguments.require_sm120,
            require_native_solver=arguments.require_native_solver,
        )
    except (ConfigError, PreflightError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
