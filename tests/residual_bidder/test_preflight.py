from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from residual_bidder.cli.preflight import PreflightError, inspect_runtime
from residual_bidder.config import BidderConfig


class _FakeTensor:
    def __init__(self, *, finite: bool = True) -> None:
        self.finite = finite
        self.grad: _FakeTensor | None = None

    def square(self) -> _FakeTensor:
        return self

    def mean(self) -> _FakeLoss:
        return _FakeLoss()


class _FakeLoss(_FakeTensor):
    def backward(self) -> None:
        assert _FakeTorch.active is not None
        for tensor in _FakeTorch.active.inputs:
            tensor.grad = _FakeTensor(finite=_FakeTorch.active.gradient_finite)


class _FakeFinite:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> _FakeFinite:
        return self

    def item(self) -> bool:
        return self.value


class _FakeCuda:
    def __init__(self, *, available: bool = True, arch_list: list[str] | None = None) -> None:
        self.available = available
        self.arch_list = arch_list or ["sm_90", "sm_120"]
        self.synchronize_calls = 0

    def is_available(self) -> bool:
        return self.available

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "Fake RTX 5090"

    def get_device_capability(self, index: int) -> tuple[int, int]:
        assert index == 0
        return (12, 0)

    def get_arch_list(self) -> list[str]:
        return list(self.arch_list)

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _FakeTorch:
    active: _FakeTorch | None = None
    __version__ = "2.8.0+cu128"
    float32 = "float32"
    version = SimpleNamespace(cuda="12.8")

    def __init__(
        self,
        *,
        cuda_available: bool = True,
        arch_list: list[str] | None = None,
        output_finite: bool = True,
        gradient_finite: bool = True,
    ) -> None:
        self.cuda = _FakeCuda(available=cuda_available, arch_list=arch_list)
        self.output_finite = output_finite
        self.gradient_finite = gradient_finite
        self.inputs: list[_FakeTensor] = []
        self.manual_seed_value: int | None = None
        _FakeTorch.active = self

    def manual_seed(self, seed: int) -> None:
        self.manual_seed_value = seed

    def randn(self, *shape: int, **kwargs: object) -> _FakeTensor:
        assert shape == (32, 32)
        assert kwargs == {"device": "cuda", "dtype": "float32", "requires_grad": True}
        tensor = _FakeTensor()
        self.inputs.append(tensor)
        return tensor

    def matmul(self, left: _FakeTensor, right: _FakeTensor) -> _FakeTensor:
        assert [left, right] == self.inputs
        return _FakeTensor(finite=self.output_finite)

    def isfinite(self, tensor: _FakeTensor) -> _FakeFinite:
        return _FakeFinite(tensor.finite)


class _FakeNativeSolver:
    def __init__(
        self,
        library_path: Path,
        *,
        available: bool = True,
        build_id: str = "fake-native-build-id",
    ) -> None:
        self.native_available = available
        self.native_build_id = build_id
        self.native_library_path = str(library_path)


@pytest.fixture
def config() -> BidderConfig:
    return BidderConfig.load(Path("configs/residual_bidder/base.yaml"))


def test_preflight_rejects_unavailable_cuda(config: BidderConfig) -> None:
    fake_torch = _FakeTorch(cuda_available=False)

    with pytest.raises(PreflightError, match="CUDA is required"):
        inspect_runtime(config, require_cuda=True, require_sm120=False, torch_module=fake_torch)


def test_preflight_rejects_missing_sm120(config: BidderConfig) -> None:
    fake_torch = _FakeTorch(arch_list=["sm_80", "sm_90"])

    with pytest.raises(PreflightError, match="sm_120"):
        inspect_runtime(config, require_cuda=True, require_sm120=True, torch_module=fake_torch)


@pytest.mark.parametrize(
    ("output_finite", "gradient_finite", "message"),
    [(False, True, "non-finite forward"), (True, False, "non-finite gradient")],
)
def test_preflight_rejects_nonfinite_gpu_probe(
    config: BidderConfig,
    output_finite: bool,
    gradient_finite: bool,
    message: str,
) -> None:
    fake_torch = _FakeTorch(
        output_finite=output_finite,
        gradient_finite=gradient_finite,
    )

    with pytest.raises(PreflightError, match=message):
        inspect_runtime(config, require_cuda=True, require_sm120=True, torch_module=fake_torch)


def test_preflight_rejects_missing_native_solver(
    config: BidderConfig, tmp_path: Path
) -> None:
    library = tmp_path / "solver.so"
    library.write_bytes(b"fake native library")

    with pytest.raises(PreflightError, match="native solver is required"):
        inspect_runtime(
            config,
            require_cuda=False,
            require_sm120=False,
            torch_module=_FakeTorch(cuda_available=False),
            solver_factory=lambda: _FakeNativeSolver(library, available=False),
            require_native_solver=True,
        )


def test_preflight_rejects_frozen_file_hash_drift(
    config: BidderConfig, tmp_path: Path
) -> None:
    drifted = tmp_path / "bid_nsfp.pt"
    drifted.write_bytes(b"not the frozen checkpoint")
    changed = replace(config, nsfp=replace(config.nsfp, path=str(drifted)))

    with pytest.raises(PreflightError, match=r"bid_nsfp\.pt.*SHA-256 mismatch"):
        inspect_runtime(
            changed,
            require_cuda=False,
            require_sm120=False,
            torch_module=_FakeTorch(cuda_available=False),
        )


def test_preflight_success_reports_capabilities_and_all_hashes(
    config: BidderConfig, tmp_path: Path
) -> None:
    library = tmp_path / "solver.so"
    library.write_bytes(b"fake native library")
    fake_torch = _FakeTorch()

    report = inspect_runtime(
        config,
        require_cuda=True,
        require_sm120=True,
        torch_module=fake_torch,
        solver_factory=lambda: _FakeNativeSolver(library),
        require_native_solver=True,
    )

    assert report["ok"] is True
    assert report["python"]
    assert report["torch"] == "2.8.0+cu128"
    assert report["compiled_cuda"] == "12.8"
    assert report["gpu_name"] == "Fake RTX 5090"
    assert report["compute_capability"] == [12, 0]
    assert report["architecture_list"] == ["sm_90", "sm_120"]
    assert report["sm_120"] is True
    assert report["native_solver"] is True
    assert report["native_solver_build_id"] == "fake-native-build-id"
    assert report["native_library_sha256"] == hashlib.sha256(
        b"fake native library"
    ).hexdigest()
    assert report["config_sha256"] == config.sha256()
    assert report["nsfp_sha256"] == config.nsfp.sha256
    assert report["play_config_sha256"] == config.play.config_sha256
    assert len(report["play_source_hashes"]) == len(config.play.source_manifest)
    assert report["play_pipeline_sha256"]
    assert fake_torch.manual_seed_value == 20260721
    assert fake_torch.cuda.synchronize_calls == 1
