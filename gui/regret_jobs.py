"""「完整复盘」分析任务的内存外（磁盘）作业仓库。

为什么用磁盘而不是进程内字典：
- 生产部署（`entrypoint.sh`）会起 N 个无状态 Python 后端，由 Caddy 轮询转发。
  创建任务的请求和轮询状态的请求可能落在不同的后端进程上，进程内字典会
  让轮询永远看不到任务。
- 所有后端（以及分析子进程）共享同一个容器的文件系统，因此「一个目录 +
  一个 JSON 文件」就是最可靠的状态载体，也天然支持进程重启后继续轮询。

目录结构：
    <job_root>/<jobId>/job.json      进度与状态（子进程反复覆写，原子替换）
    <job_root>/<jobId>/result.json   最终结果（只在完成时写一次）
    <job_root>/<jobId>/record.json   提交上来的复盘记录
    <job_root>/<jobId>/worker.log    子进程 stdout/stderr

只依赖标准库，方便被任何模块导入而不引入 torch / 求解器依赖。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

# 作业 id 会被拼进文件路径，因此必须是受限字符集。
_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

DEFAULT_JOB_ROOT = Path(
    os.environ.get("SPADES_REGRET_JOB_ROOT")
    or (Path(tempfile.gettempdir()) / "spades-regret-jobs")
)

# 一个作业目录最多保留多久（秒）。超时目录会在下一次创建任务时被清理。
DEFAULT_JOB_TTL_SECONDS = 24 * 60 * 60

# 同时最多保留多少个已完成作业目录。
MAX_RETAINED_JOBS = 24


class JobNotFoundError(KeyError):
    """请求的作业 id 不存在（或已被清理）。"""


def job_root() -> Path:
    """返回作业根目录（读取环境变量以便测试与部署覆盖）。"""
    override = os.environ.get("SPADES_REGRET_JOB_ROOT")
    root = Path(override) if override else DEFAULT_JOB_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_job_id() -> str:
    """生成一个新的、路径安全的任务 id。"""
    return uuid.uuid4().hex


def validate_job_id(job_id: Any) -> str:
    """校验任务 id 只包含十六进制字符，避免目录穿越。"""
    if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
        raise JobNotFoundError(f"非法的 jobId: {job_id!r}")
    return job_id


def job_dir(job_id: str, root: Path | None = None) -> Path:
    """返回任务目录（不创建）。"""
    base = root if root is not None else job_root()
    return base / validate_job_id(job_id)


def _atomic_write_json(path: Path, payload: Any) -> None:
    """先写临时文件再 rename，保证读端永远看到完整的 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def _read_json(path: Path) -> Any | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def write_job_state(job_id: str, payload: dict[str, Any], root: Path | None = None) -> None:
    """覆写 `job.json`。"""
    _atomic_write_json(job_dir(job_id, root) / "job.json", payload)


def read_job_state(job_id: str, root: Path | None = None) -> dict[str, Any] | None:
    """读取 `job.json`；不存在或损坏时返回 None。"""
    return _read_json(job_dir(job_id, root) / "job.json")


def write_job_result(job_id: str, payload: dict[str, Any], root: Path | None = None) -> None:
    """写出最终结果 `result.json`（只写一次）。"""
    _atomic_write_json(job_dir(job_id, root) / "result.json", payload)


def read_job_result(job_id: str, root: Path | None = None) -> dict[str, Any] | None:
    """读取最终结果；尚未完成时返回 None。"""
    return _read_json(job_dir(job_id, root) / "result.json")


def write_job_record(job_id: str, record: Any, root: Path | None = None) -> None:
    """持久化提交上来的复盘记录，便于失败后复查。"""
    _atomic_write_json(job_dir(job_id, root) / "record.json", record)


def read_job_record(job_id: str, root: Path | None = None) -> Any | None:
    """读取提交上来的复盘记录。"""
    return _read_json(job_dir(job_id, root) / "record.json")


def worker_log_path(job_id: str, root: Path | None = None) -> Path:
    """返回子进程日志文件路径（调用方负责创建父目录）。"""
    directory = job_dir(job_id, root)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "worker.log"


def create_job(root: Path | None = None) -> str:
    """创建任务目录并返回新的 job id。"""
    prune_jobs(root)
    job_id = new_job_id()
    job_dir(job_id, root).mkdir(parents=True, exist_ok=True)
    write_job_state(
        job_id,
        {
            "jobId": job_id,
            "status": "queued",
            "createdAt": time.time(),
            "updatedAt": time.time(),
            "progress": {"done": 0, "total": 0, "current": None},
            "error": None,
        },
        root,
    )
    return job_id


def set_job_status(
    job_id: str,
    status: str,
    *,
    progress: dict[str, Any] | None = None,
    error: str | None = None,
    root: Path | None = None,
    job_extra: dict[str, Any] | None = None,
) -> None:
    """更新任务状态（读-改-写，保留 createdAt 等已有字段）。"""
    current = read_job_state(job_id, root) or {"jobId": job_id, "createdAt": time.time()}
    current["status"] = status
    current["updatedAt"] = time.time()
    if progress is not None:
        current["progress"] = progress
    if error is not None:
        current["error"] = error
    if job_extra:
        current.update(job_extra)
    write_job_state(job_id, current, root)


def prune_jobs(root: Path | None = None, ttl_seconds: float = DEFAULT_JOB_TTL_SECONDS) -> None:
    """清理超期目录，并把保留数量压到上限以内。"""
    base = root if root is not None else job_root()
    if not base.exists():
        return
    try:
        entries = [path for path in base.iterdir() if path.is_dir()]
    except OSError:
        return

    now = time.time()
    alive: list[tuple[float, Path]] = []
    for path in entries:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        state = _read_json(path / "job.json")
        status = (state or {}).get("status")
        # 正在跑的任务无论多旧都不清理。
        if status in {"queued", "running"} and now - mtime < ttl_seconds:
            alive.append((mtime, path))
            continue
        if now - mtime > ttl_seconds:
            shutil.rmtree(path, ignore_errors=True)
            continue
        alive.append((mtime, path))

    alive.sort(key=lambda item: item[0], reverse=True)
    for _, path in alive[MAX_RETAINED_JOBS:]:
        shutil.rmtree(path, ignore_errors=True)


def job_snapshot(job_id: str, root: Path | None = None) -> dict[str, Any]:
    """汇总任务状态与（可能已完成的）结果，供 HTTP 层直接返回。"""
    state = read_job_state(job_id, root)
    if state is None:
        raise JobNotFoundError(f"未知的 jobId: {job_id}")

    snapshot: dict[str, Any] = {
        "jobId": job_id,
        "status": state.get("status", "unknown"),
        "createdAt": state.get("createdAt"),
        "updatedAt": state.get("updatedAt"),
        "progress": state.get("progress") or {"done": 0, "total": 0, "current": None},
        "error": state.get("error"),
    }
    ai = state.get("ai")
    if ai is not None:
        snapshot["ai"] = ai
    if snapshot["status"] == "done":
        result = read_job_result(job_id, root)
        if result is not None:
            snapshot["result"] = result
    return snapshot
