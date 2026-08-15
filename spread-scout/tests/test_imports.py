"""Deployment guard: app.py must import with no help from the test harness.

This exists because a green test suite once sat next to a broken deploy. The
smoke harness did `sys.path.insert(0, ".../spread-scout")` before loading the
app, so `from strategy import ...` always resolved locally — while Streamlit
Cloud, which runs the script from the repo root without adding the script's
own directory to sys.path, raised ModuleNotFoundError.

Run: python3 spread-scout/tests/test_imports.py
"""

import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"
REPO = APP.parents[1]
FAIL = []


def imports_cleanly(cwd: Path) -> tuple[bool, str]:
    """Execute everything above the first Streamlit call, in a fresh
    interpreter, with PYTHONPATH deliberately unset."""
    head = APP.read_text().split("inject_css()")[0]
    # __file__ must be set, exactly as Streamlit sets it when it runs the
    # script — the path bootstrap depends on it.
    prog = (f"g = {{'__file__': {str(APP)!r}}}\n"
            f"exec(compile({head!r}, {str(APP)!r}, 'exec'), g)")
    r = subprocess.run(
        [sys.executable, "-c", prog],
        capture_output=True, text=True, cwd=str(cwd),
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
    )
    err = (r.stderr or "").strip()
    # any traceback at all is a failure here, not just an import error
    return (r.returncode == 0), (err.splitlines()[-1] if err else "")


for label, cwd in (("repo root", REPO), ("a foreign cwd", Path("/tmp")),
                   ("the app's own dir", APP.parent)):
    ok, msg = imports_cleanly(cwd)
    print(f"  {'PASS' if ok else 'FAIL'}  imports from {label}"
          + (f" -> {msg}" if not ok else ""))
    if not ok:
        FAIL.append(label)

# every name app.py imports must actually exist in strategy.py
sys.path.insert(0, str(APP.parent))
import strategy  # noqa: E402

block = APP.read_text().split("from strategy import (")[1].split(")")[0]
names = [n.strip().rstrip(",") for n in block.replace("# noqa: E402", "")
         .replace("\n", " ").split() if n.strip().rstrip(",")]
missing = [n for n in names if not hasattr(strategy, n.rstrip(","))]
print(f"  {'PASS' if not missing else 'FAIL'}  every imported name exists in "
      f"strategy.py ({len(names)} names)"
      + (f" -> missing {missing}" if missing else ""))
if missing:
    FAIL.append("missing names")

print()
if FAIL:
    print(f"FAILED: {FAIL}")
    sys.exit(1)
print("IMPORT GUARD PASSES")
