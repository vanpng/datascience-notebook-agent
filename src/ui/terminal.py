"""Rich-based terminal UI for the DS Notebook Agent."""
from __future__ import annotations

import argparse
import sys
import uuid

import httpx
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

console = Console()

WELCOME = """
# DS Notebook Agent — Terminal UI

Type a data science query to generate and run a notebook cell.
Commands:
  **:history**  — show executed cells in this session
  **:context**  — print current DataFrame schemas
  **:clear**    — reset the session (new session ID)
  **:quit**     — exit
"""


class AgentClient:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self.session_id: str | None = None
        self._http = httpx.Client(timeout=300)   # long timeout for inference

    def new_session(self) -> str:
        sid = str(uuid.uuid4())
        r = self._http.post(f"{self.api_url}/sessions", json={"session_id": sid})
        r.raise_for_status()
        self.session_id = sid
        return sid

    def get_session(self) -> dict:
        r = self._http.get(f"{self.api_url}/sessions/{self.session_id}")
        r.raise_for_status()
        return r.json()

    def add_cell(self, source: str, stdout: str = "", stderr: str = "", success: bool = True):
        r = self._http.post(
            f"{self.api_url}/sessions/{self.session_id}/cells",
            json={"source": source, "stdout": stdout, "stderr": stderr, "success": success},
        )
        r.raise_for_status()

    def query(self, text: str) -> dict:
        r = self._http.post(
            f"{self.api_url}/sessions/{self.session_id}/query",
            json={"query": text},
        )
        r.raise_for_status()
        return r.json()

    def close(self):
        self._http.close()


# ── display helpers ───────────────────────────────────────────────────────────

def _print_code(code: str, title: str = "Generated Cell"):
    console.print(
        Panel(
            Syntax(code, "python", theme="monokai", line_numbers=True),
            title=f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
        )
    )


def _print_output(stdout: str, stderr: str, success: bool):
    if stdout:
        console.print(Panel(stdout.strip(), title="[green]stdout[/green]", border_style="green"))
    if stderr and not success:
        console.print(Panel(stderr.strip(), title="[red]stderr[/red]", border_style="red"))


def _print_history(session: dict):
    cells = session.get("notebook_cells", [])
    if not cells:
        console.print("[dim]No cells in history.[/dim]")
        return
    for c in cells:
        icon = "✓" if c["success"] else "✗"
        color = "green" if c["success"] else "red"
        console.print(Rule(f"[{color}]{icon} Cell {c['cell_id']}[/{color}]"))
        _print_code(c["source"], title=f"Cell {c['cell_id']}")
        if c.get("stdout"):
            console.print(Text(c["stdout"].strip(), style="dim"))


def _print_schemas(session: dict):
    schemas = session.get("dataframe_schemas", [])
    if not schemas:
        console.print("[dim]No DataFrames detected yet.[/dim]")
        return
    for s in schemas:
        console.print(
            Panel(
                f"columns: {s['columns']}\ndtypes:  {s['dtypes']}",
                title=f"[bold yellow]{s['name']}[/bold yellow]",
                border_style="yellow",
            )
        )


# ── main loop ─────────────────────────────────────────────────────────────────

def _run(api_url: str):
    console.print(Markdown(WELCOME))
    client = AgentClient(api_url)

    try:
        client.new_session()
        console.print(f"[dim]Session: {client.session_id}[/dim]")
    except Exception as exc:
        console.print(f"[bold red]Cannot reach agent API at {api_url}: {exc}[/bold red]")
        sys.exit(1)

    while True:
        try:
            raw = Prompt.ask("\n[bold magenta]Query[/bold magenta]").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not raw:
            continue

        if raw == ":quit":
            break
        if raw == ":clear":
            client.new_session()
            console.print(f"[dim]New session: {client.session_id}[/dim]")
            continue
        if raw == ":history":
            _print_history(client.get_session())
            continue
        if raw == ":context":
            _print_schemas(client.get_session())
            continue

        # ── send query to agent ───────────────────────────────────────────────
        with console.status("[bold cyan]Agent thinking…[/bold cyan]", spinner="dots"):
            try:
                resp = client.query(raw)
            except httpx.HTTPError as exc:
                console.print(f"[red]HTTP error: {exc}[/red]")
                continue

        status = resp.get("status")
        debug_n = resp.get("debug_attempts", 0)

        if status == "done":
            console.print(f"\n[green]✓ Done[/green]"
                          + (f" (debug attempts: {debug_n})" if debug_n else ""))
            if resp.get("final_code"):
                _print_code(resp["final_code"])
            _print_output(resp.get("stdout", ""), resp.get("stderr", ""), success=True)
        else:
            console.print(f"\n[red]✗ Failed[/red] (debug attempts: {debug_n})")
            if resp.get("agent_error"):
                console.print(Panel(resp["agent_error"], title="[red]Error[/red]",
                                    border_style="red"))
            if resp.get("stderr"):
                console.print(Panel(resp["stderr"], title="[red]Last stderr[/red]",
                                    border_style="red"))

    client.close()
    console.print("\n[dim]Goodbye.[/dim]")


def main():
    parser = argparse.ArgumentParser(description="DS Notebook Agent Terminal UI")
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="FastAPI agent server URL",
    )
    args = parser.parse_args()
    _run(args.api_url)


if __name__ == "__main__":
    main()
