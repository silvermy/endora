"""
tests/test_ha_panel_docs.py

The Home Assistant sidebar panel under docs/homeassistant/.

Nothing in this project executes that JavaScript, so the one thing worth
guarding is the join between the two files a user copies: panel_custom's
`name:` is the custom element tag, and HA instantiates exactly that tag. If
it does not match the customElements.define call, the panel loads and renders
nothing at all — no error, just an empty page — and there is no way to tell
from the HA side what went wrong.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs" / "homeassistant"
JS = DOCS / "endora-console.js"
README = DOCS / "README.md"


def test_the_files_exist():
    assert JS.is_file(), JS
    assert README.is_file(), README


def _defined_tag() -> str:
    m = re.search(r'customElements\.define\(\s*"([^"]+)"', JS.read_text())
    assert m, "no customElements.define call in the panel JS"
    return m.group(1)


def _panel_name() -> str:
    m = re.search(r"^\s*-\s*name:\s*(\S+)\s*$", README.read_text(), re.M)
    assert m, "no panel_custom `name:` in the README snippet"
    return m.group(1)


def test_panel_name_matches_the_custom_element_tag():
    assert _panel_name() == _defined_tag()


def test_the_tag_is_a_valid_custom_element_name():
    # Custom elements must contain a hyphen; HA's own docs example is
    # `my-panel` for this reason. A name like "HTTP Link" never registers.
    tag = _defined_tag()
    assert "-" in tag, tag
    assert tag == tag.lower(), tag
    assert " " not in tag, tag


def test_module_url_points_at_the_file_we_ship():
    m = re.search(r"module_url:\s*(\S+)", README.read_text())
    assert m, "no module_url in the README snippet"
    assert m.group(1).split("?")[0].endswith(JS.name), m.group(1)


def test_target_is_an_absolute_http_url():
    # Relative or scheme-less would resolve against HA's own HTTPS origin —
    # the exact failure this panel exists to avoid.
    m = re.search(r'const TARGET\s*=\s*"([^"]+)"', JS.read_text())
    assert m, "no TARGET in the panel JS"
    assert re.match(r"^https?://[^/]+:\d+/$", m.group(1)), m.group(1)


def test_the_panel_behaves_correctly_when_executed():
    """Run docs/homeassistant/endora-console.js against a stub DOM.

    Static checks cannot see the bug that motivated this: window.open()
    returns null when passed "noopener", so testing its return value for
    success is always false and the panel claims the popup was blocked even
    when the tab opened. Only executing it shows that.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; JS panel behaviour unverified")
    runner = Path(__file__).resolve().parent / "js" / "run_panel_tests.js"
    proc = subprocess.run([node, str(runner)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
