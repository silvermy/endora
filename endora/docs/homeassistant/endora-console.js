// Endora — Home Assistant sidebar panel that opens the debug console in a
// new browser tab.
//
// Install: copy to /config/www/endora-console.js, add the panel_custom block
// from README.md to configuration.yaml, restart Home Assistant.
//
// TARGET is wherever Endora's debug server listens, logged at startup as
// "Debug stream: http://<host>:<port>/".
const TARGET = "http://10.0.0.141:8765/";

// Home Assistant renders sidebar entries in-page and offers no way to make
// one open an external URL, so the panel does it instead: panel_custom with
// embed_iframe false loads this module into HA's own frontend, where it can
// call window.open. Embedding was never an option — HA is HTTPS and the
// console HTTP, and mixed-content blocking rejects an HTTP iframe. It does
// not reject top-level navigation, which is what this is.
class EndoraConsole extends HTMLElement {
  connectedCallback() {
    // "noopener" is deliberately absent: passed, window.open() returns null
    // whether or not it succeeded, so the branch below becomes a coin flip.
    // Clearing .opener gives the same isolation and a usable return value.
    let win = null;
    try {
      win = window.open(TARGET, "_blank");
      if (win) win.opener = null;
    } catch (e) { /* treat as blocked */ }

    this._render(!!win);
    if (win) this._goHome();
  }

  // Both actions are always offered, and "back" comes first.
  //
  // This panel draws its own content with no Home Assistant toolbar, so it
  // has no menu button. On a desktop the sidebar is on screen anyway; in the
  // iOS app it is hidden behind that missing button, which left no way out
  // of this page at all. On mobile window.open is also usually refused, so
  // the panel takes the "blocked" branch and deliberately does not navigate
  // away — correct on a desktop, a dead end on a phone.
  _render(opened) {
    const btn = "display:inline-block;padding:12px 18px;margin:0 12px 12px 0;" +
                "border-radius:8px;text-decoration:none;font-size:15px";
    this.innerHTML = `
      <div style="padding:24px;color:var(--primary-text-color,#212121)">
        <p>${opened ? "Endora opened in a new tab."
                    : "Open the Endora debug console:"}</p>
        <p>
          <a href="#" id="endora-back"
             style="${btn};background:var(--primary-color,#03a9f4);color:#fff"
            >&#8592; Home Assistant</a>
          <a href="${TARGET}" target="_blank" rel="noopener"
             style="${btn};border:1px solid var(--primary-color,#03a9f4);
                    color:var(--primary-color,#03a9f4)"
            >Endora console &#8599;</a>
        </p>
      </div>`;

    this.querySelector("#endora-back").addEventListener("click", (e) => {
      e.preventDefault();
      // history.back() is the fallback rather than the first choice: it
      // leaves Home Assistant entirely if this panel was opened from a cold
      // start, where there is nothing to go back to.
      if (!this._goHome()) history.back();
    });

    this._addMenuButton();
  }

  // Home Assistant's own hamburger, so the sidebar is reachable from here.
  //
  // Built-in panels render a toolbar containing it; this one draws its own
  // markup and had none, which on a desktop is invisible (the sidebar is
  // always on screen) and on a phone means the sidebar cannot be opened at
  // all. Best-effort: the element belongs to the frontend, not to us, so if
  // it is not registered under this name we simply do without — the button
  // above is the guaranteed way out either way.
  _addMenuButton() {
    if (!customElements.get("ha-menu-button")) return;
    try {
      const bar = document.createElement("div");
      bar.style.cssText = "padding:8px 8px 0";
      const menu = document.createElement("ha-menu-button");
      menu.hass = this.hass;
      menu.narrow = this.narrow !== undefined ? this.narrow : true;
      bar.appendChild(menu);
      this.insertBefore(bar, this.firstChild);
    } catch (e) { /* cosmetic only */ }
  }

  // Send HA to a dashboard so it never rests on this panel; otherwise every
  // reload that restores this route opens another tab. Returns whether it
  // navigated, so the Back button can fall back to history.back().
  _goHome() {
    const cfg = (this.panel && this.panel.config) || {};
    const fallback = this.hass && this.hass.defaultPanel;
    const home = cfg.home_path || (fallback ? "/" + fallback : null);

    // Never guess a dashboard. hass.defaultPanel can be empty even when hass
    // is set, and HA answers a route with no dashboard behind it by spinning
    // forever — which reads as this panel having failed. Staying put is
    // always safe: it has rendered, and it carries a link.
    if (!home) return false;

    // Deferred, and fired on window. Dispatched inline from
    // connectedCallback the event does not reach HA's router; window is
    // where HA's own navigate() helper fires it. replaceState rather than
    // push, so Back does not land here and open another tab, and
    // history.state carries HA's routing state across.
    const go = () => {
      history.replaceState(history.state, "", home);
      window.dispatchEvent(new CustomEvent("location-changed",
        { detail: { replace: true }, bubbles: true, composed: true }));
    };

    // Immediately, in the same tick as the window.open that triggered it.
    //
    // This used to be deferred by a tick and nothing else, on the theory
    // that an inline dispatch does not reach Home Assistant's router. That
    // theory came from debugging a spinner whose real cause turned out to be
    // a dashboard path that did not exist, so it was never the reason — and
    // deferring is precisely what breaks iOS, where window.open hands off to
    // Safari and the webview is suspended before the timer can run. Going
    // now, while the page is still in the foreground, is what makes the
    // panel vanish instead of sitting there waiting to be dismissed.
    go();

    // Backstops, in case a host does need the deferral after all. Every path
    // is idempotent: replaceState to the same path changes nothing.
    setTimeout(go, 0);

    // …and again when the app comes back to the foreground.
    //
    // On iOS, window.open hands off to Safari and the Home Assistant app's
    // webview is backgrounded immediately — which suspends JavaScript, so
    // the timer above never fires. Returning to the app then shows a panel
    // that should have navigated away and did not. Re-running when the page
    // becomes visible completes the hop that was dropped; doing it twice is
    // harmless, since replaceState to the same path is idempotent.
    const onReturn = () => {
      if (!document.hidden) go();
    };
    document.addEventListener("visibilitychange", onReturn, { once: true });
    window.addEventListener("pageshow", go, { once: true });
    return true;
  }
}

// The tag must match panel_custom's `name:`. Guarded because define() throws
// on a name that is already registered.
if (!customElements.get("endora-console")) {
  customElements.define("endora-console", EndoraConsole);
}
