"""
IPython magic extension — integrates the DS notebook agent into JupyterLab.

Usage (in any notebook cell):
    %load_ext src.agent.magic          # once per kernel session
    %agent build a classification model for this data
"""
from __future__ import annotations

import io
import os
import textwrap
from typing import Any

import httpx
from IPython.core.magic import Magics, line_magic, magics_class
from IPython.display import Markdown, display

API_URL = os.getenv("AGENT_API_URL", "http://127.0.0.1:8000")

# Cells containing these prefixes are skipped when syncing to the agent
_SKIP_PREFIXES = ("%", "!", "get_ipython", "# %%", "# In[")

# ── per-kernel stdout capture ─────────────────────────────────────────────────
# Maps IPython execution_count → captured stdout text for that cell.
# Populated by the pre/post_run_cell hooks registered in load_ipython_extension.
_cell_outputs: dict[int, str] = {}
_capture_state: dict[str, Any] = {"buf": None, "real_stdout": None}


class _StdoutTee:
    """Forwards writes to the real stdout AND captures them into a StringIO buffer.

    Using __slots__ + explicit object.__getattribute__ avoids infinite recursion
    when delegating arbitrary attributes of IPython's OutStream to _real.
    """
    __slots__ = ("_real", "_buf")

    def __init__(self, real: Any, buf: io.StringIO) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_buf", buf)

    def write(self, s: str) -> int:
        self._real.write(s)
        self._buf.write(s)
        return len(s)

    def flush(self) -> None:
        self._real.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_real", "_buf"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_real"), name, value)


def _pre_run_cell(info: Any) -> None:
    """Wrap sys.stdout in a Tee before each cell so we can capture its output."""
    import sys
    if isinstance(sys.stdout, _StdoutTee):
        return  # already wrapped (re-entrant call from %run etc.)
    buf = io.StringIO()
    _capture_state["buf"] = buf
    _capture_state["real_stdout"] = sys.stdout
    sys.stdout = _StdoutTee(sys.stdout, buf)


def _post_run_cell(result: Any) -> None:
    """Restore sys.stdout and persist captured text + expression repr."""
    import sys
    if _capture_state["real_stdout"] is not None:
        sys.stdout = _capture_state["real_stdout"]
        _capture_state["real_stdout"] = None

    captured = ""
    if _capture_state["buf"] is not None:
        captured = _capture_state["buf"].getvalue()
        _capture_state["buf"] = None

    # Also grab the repr of Out[n] (expression result, e.g. df.head())
    expr = getattr(result, "result", None)
    if expr is not None:
        r = repr(expr)
        if r not in ("None", ""):
            captured = (captured.rstrip() + "\n" + r).strip() if captured.strip() else r

    ec = getattr(result, "execution_count", None)
    if ec is not None and captured:
        _cell_outputs[ec] = captured


@magics_class
class DSAgentMagic(Magics):
    def __init__(self, shell, **kwargs):
        super().__init__(shell, **kwargs)
        self._client = httpx.Client(timeout=300, base_url=API_URL)
        self._session_id: str | None = None
        self._synced_up_to: int = 0       # index into input_hist_parsed

    # ── session management ────────────────────────────────────────────────────

    def _session(self) -> str:
        if self._session_id is None:
            r = self._client.post("/sessions", json={})
            r.raise_for_status()
            self._session_id = r.json()["session_id"]
        return self._session_id

    def _sync_cells(self) -> None:
        """Push any new executed cells (with their captured stdout) to the agent session.

        input_hist_parsed[i] corresponds to IPython execution count i, so we use
        the loop index to look up the stdout captured by _post_run_cell.
        """
        history = self.shell.history_manager.input_hist_parsed
        sid = self._session()

        start = self._synced_up_to
        for i, src in enumerate(history[start:], start=start):
            self._synced_up_to = i + 1
            src = src.strip()
            if not src or any(src.startswith(p) for p in _SKIP_PREFIXES):
                continue
            stdout = _cell_outputs.get(i, "")
            self._client.post(
                f"/sessions/{sid}/cells",
                json={"source": src, "stdout": stdout, "stderr": "", "success": True},
            ).raise_for_status()

    # ── magics ────────────────────────────────────────────────────────────────

    @line_magic
    def agent(self, line: str) -> None:
        """Send a natural-language query to the DS agent.

        The generated code is inserted as the next input cell.

        Example::
            %agent show survival rate by class as a bar chart
        """
        query = line.strip()
        if not query:
            print("Usage:  %agent <your query>")
            return

        self._sync_cells()
        sid = self._session()

        display(Markdown(f"🤖 **Agent** — *{query}*"))

        try:
            resp = self._client.post(
                f"/sessions/{sid}/query", json={"query": query, "execute": False}
            ).raise_for_status().json()
        except Exception as exc:
            display(Markdown(f"❌ **API error**: `{exc}`"))
            return

        status = resp.get("status")
        code   = resp.get("final_code", "")
        stdout = resp.get("stdout", "")
        err    = resp.get("agent_error", "")
        debug  = resp.get("debug_attempts", 0)

        if status == "done":
            badge = "✅ Done" + (f" *(debug attempts: {debug})*" if debug else "")
            display(Markdown(badge))
            if stdout:
                display(Markdown(f"**Output preview**\n```\n{stdout.strip()[:500]}\n```"))
            if code:
                # Insert the generated code as the next editable cell
                self.shell.set_next_input(code, replace=False)
        else:
            display(Markdown(f"❌ **Failed** after {debug} debug attempts"))
            if err:
                display(Markdown(f"```\n{textwrap.shorten(err, 800)}\n```"))

    @line_magic
    def agent_reset(self, line: str) -> None:
        """Start a fresh agent session (keeps the kernel namespace intact)."""
        if self._session_id:
            try:
                self._client.delete(f"/sessions/{self._session_id}")
            except Exception:
                pass
        self._session_id = None
        self._synced_up_to = 0
        display(Markdown("🔄 Agent session reset."))

    @line_magic
    def agent_status(self, line: str) -> None:
        """Print the current session ID and how many cells have been synced."""
        display(Markdown(
            f"**Session**: `{self._session_id or 'not started'}`  \n"
            f"**Cells synced**: {self._synced_up_to}"
        ))


def load_ipython_extension(ipython) -> None:
    ipython.register_magics(DSAgentMagic)
    ipython.events.register("pre_run_cell", _pre_run_cell)
    ipython.events.register("post_run_cell", _post_run_cell)
    print("DS Agent magic loaded.  Use %agent <query> to generate cells.")
