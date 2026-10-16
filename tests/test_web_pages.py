"""Static checks on the web pages: no missing files, no element ids the scripts
look for that the page does not have, and every module parses. The browser
would fail silently on all three."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
PAGES = {"index.html": ["app.js"], "runs.html": ["runs.js"], "results.html": ["results.js"]}


def _local_imports(script: Path) -> set[str]:
    text = script.read_text(encoding="utf-8")
    return set(re.findall(r"""(?:from|import\()\s*["']\./([\w.-]+)["']""", text))


@pytest.mark.parametrize("page", PAGES)
def test_every_local_file_a_page_loads_exists(page):
    html = (WEB / page).read_text(encoding="utf-8")
    refs = set(re.findall(r'(?:src|href)="([\w.-]+\.(?:js|css|html))"', html))
    seen, queue = set(), [r for r in refs if r.endswith(".js")]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        queue.extend(_local_imports(WEB / name))
    for name in refs | seen:
        assert (WEB / name).is_file(), f"{page} needs {name}"


@pytest.mark.parametrize("page,scripts", PAGES.items())
def test_scripts_only_look_up_ids_the_page_has(page, scripts):
    html = (WEB / page).read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([\w-]+)"', html))
    for script in scripts:
        text = (WEB / script).read_text(encoding="utf-8")
        wanted = set(re.findall(r"""\$\(\s*["']([\w-]+)["']\s*\)""", text))
        wanted |= set(re.findall(r"""getElementById\(\s*["']([\w-]+)["']\s*\)""", text))
        missing = sorted(wanted - ids)
        assert not missing, f"{script} looks up ids {page} does not have: {missing}"


@pytest.mark.parametrize("script", sorted(p.name for p in WEB.glob("*.js")))
def test_every_module_parses(script):
    """A syntax error (say, a name declared twice) stops the whole module loading."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    out = subprocess.run([node, "--input-type=module", "--check"], input=(WEB / script).read_text(encoding="utf-8"),
                         capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr
