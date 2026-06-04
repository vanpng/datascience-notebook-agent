"""Isolated code execution sandbox.

Local (dev): subprocess with resource limits via resource module.
Cluster (Tillicum): Apptainer container with --containall + memory cap.
"""
from __future__ import annotations

import asyncio
import os
import resource
import sys
import tempfile
from pathlib import Path

from src.agent.schemas import ExecutionResult

TIMEOUT_S: int = int(os.getenv("SANDBOX_TIMEOUT", "60"))
MEMORY_MB: int = int(os.getenv("SANDBOX_MEMORY_MB", "2048"))
USE_APPTAINER: bool = os.getenv("USE_APPTAINER", "false").lower() == "true"
APPTAINER_SIF: str = os.getenv("APPTAINER_SIF", "sandbox/ds_agent.sif")

_CELL_SEPARATOR = "\n# ── prior cell ──\n"


def _memory_limit_preexec(memory_mb: int):
    """Set virtual memory limit before exec (Linux only)."""
    if sys.platform.startswith("linux"):
        limit_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


_SUPPRESS_OUTPUT = """\
import sys as _sys, io as _io
_sys.stdout = _io.StringIO()
_sys.stderr = _io.StringIO()
"""
_RESTORE_OUTPUT = """\
_sys.stdout = _sys.__stdout__
_sys.stderr = _sys.__stderr__
"""


def _build_script(prior_sources: list[str], code: str) -> str:
    """Replay prior cells (stdout suppressed) then run the new cell normally."""
    if not prior_sources:
        return code
    prior_block = _CELL_SEPARATOR.join(src.strip() for src in prior_sources if src.strip())
    return f"{_SUPPRESS_OUTPUT}{prior_block}\n{_RESTORE_OUTPUT}\n# ── new cell ──\n{code}"


async def execute_code(
    code: str,
    prior_sources: list[str] | None = None,
    timeout: int = TIMEOUT_S,
    memory_mb: int = MEMORY_MB,
) -> ExecutionResult:
    """Execute *code* in an isolated subprocess, replaying *prior_sources* first."""
    script = _build_script(prior_sources or [], code)
    if USE_APPTAINER:
        return await _execute_apptainer(script, timeout, memory_mb)
    return await _execute_subprocess(script, timeout, memory_mb)


async def _execute_subprocess(
    script: str,
    timeout: int,
    memory_mb: int,
) -> ExecutionResult:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
            preexec_fn=lambda: _memory_limit_preexec(memory_mb),
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ExecutionResult(
                stdout="", stderr=f"TimeoutError: cell exceeded {timeout}s",
                success=False, timed_out=True,
            )
    finally:
        Path(script_path).unlink(missing_ok=True)

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    success = proc.returncode == 0
    memory_exceeded = "MemoryError" in stderr or "Cannot allocate memory" in stderr

    return ExecutionResult(
        stdout=stdout, stderr=stderr,
        success=success, memory_exceeded=memory_exceeded,
    )


async def _execute_apptainer(
    script: str,
    timeout: int,
    memory_mb: int,
) -> ExecutionResult:
    """Run cell inside an Apptainer container (Tillicum cluster path)."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script)
        script_path = f.name

    cmd = [
        "apptainer", "exec",
        "--containall",
        "--bind", f"{script_path}:{script_path}",
        APPTAINER_SIF,
        "python3", script_path,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ExecutionResult(
                stdout="", stderr=f"TimeoutError: cell exceeded {timeout}s",
                success=False, timed_out=True,
            )
    finally:
        Path(script_path).unlink(missing_ok=True)

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    return ExecutionResult(
        stdout=stdout, stderr=stderr,
        success=proc.returncode == 0,
        memory_exceeded="MemoryError" in stderr,
    )
