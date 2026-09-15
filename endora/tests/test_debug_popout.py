"""
tests/test_debug_popout.py

The "Open in new tab" jump from Home Assistant's sidebar.

HA renders an add-on's ingress panel in an iframe under its own HTTPS origin
and offers no way to make a sidebar entry open an external URL, so the panel
itself carries the jump. The link has to be absolute and has to name the
add-on's own host and port — served from HA's origin, a relative href
resolves back into HA.

A top-level link from https to http is allowed; it is embedded subresources
that mixed-content blocking stops. That distinction is the whole reason this
works, and the reason the ingress panel serves the UI itself rather than
framing the http port.
"""
import re

from cameras import debug_server


def _page() -> str:
    """The rendered console page, substitutions applied."""
    import json
    return (debug_server._HTML_TEMPLATE
            .replace("__PARAMS_JSON__", json.dumps(debug_server._PARAMS))
            .replace("__TOGGLES_JSON__", json.dumps(debug_server._TOGGLES))
            .replace("__JOY_PARAMS_JSON__", json.dumps(debug_server._JOY_PARAMS))
            .replace("__DIRECT_URL__", debug_server._direct_url()))


def test_direct_url_is_absolute_with_host_and_port():
    debug_server.set_host_info("10.0.0.142", 8765)
    assert debug_server._direct_url() == "http://10.0.0.142:8765/"


def test_direct_url_follows_a_configured_port():
    debug_server.set_host_info("10.0.0.142", 9999)
    assert debug_server._direct_url().endswith(":9999/")


def test_page_carries_the_absolute_link_and_no_placeholder():
    debug_server.set_host_info("10.0.0.142", 8765)
    page = _page()
    assert "__DIRECT_URL__" not in page, "placeholder left unsubstituted"
    assert 'href="http://10.0.0.142:8765/"' in page


def test_link_opens_a_new_tab_safely():
    debug_server.set_host_info("10.0.0.142", 8765)
    page = _page()
    anchor = re.search(r'<a id="popout"[^>]*>', page)
    assert anchor, "pop-out anchor missing"
    tag = anchor.group(0)
    assert 'target="_blank"' in tag, tag
    # Without noopener the opened tab can reach back through window.opener.
    assert 'rel="noopener"' in tag, tag


def test_link_starts_hidden():
    # It is only meaningful inside the ingress frame; on the direct console it
    # would open a second copy of the page already on screen. initPopout()
    # reveals it when framed.
    debug_server.set_host_info("10.0.0.142", 8765)
    page = _page()
    anchor = re.search(r'<a id="popout"[^>]*>', page).group(0)
    assert "hidden" in anchor, anchor
    assert "initPopout()" in page
    assert "window.self !== window.top" in page
