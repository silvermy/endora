// Endora — Home Assistant sidebar panel that opens the debug console in a
// new browser tab.
//
// Install: copy to /config/www/endora-console.js, add the panel_custom block
// from README.md to configuration.yaml, restart Home Assistant.
//
// TARGET is wherever Endora's debug server listens, logged at startup as
// "Debug stream: http://<host>:<port>/".
const TARGET = "http://10.0.0.142:8765/";

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

    this.innerHTML = `
      <div style="padding:24px;color:var(--primary-text-color,#212121)">
        <p>${win ? "Endora opened in a new tab."
                 : "Your browser blocked the new tab."}</p>
        <p><a href="${TARGET}" target="_blank" rel="noopener"
              style="color:var(--primary-color,#03a9f4)"
          >Open the Endora debug console &#8599;</a></p>
      </div>`;

    if (win) this._goHome();
  }

  // Send HA to a dashboard so it never rests on this panel; otherwise every
  // reload that restores this route opens another tab.
  _goHome() {
    const cfg = (this.panel && this.panel.config) || {};
    const fallback = this.hass && this.hass.defaultPanel;
    const home = cfg.home_path || (fallback ? "/" + fallback : null);

    // Never guess a dashboard. hass.defaultPanel can be empty even when hass
    // is set, and HA answers a route with no dashboard behind it by spinning
    // forever — which reads as this panel having failed. Staying put is
    // always safe: it has rendered, and it carries a link.
    if (!home) return;

    // Deferred, and fired on window. Dispatched inline from
    // connectedCallback the event does not reach HA's router; window is
    // where HA's own navigate() helper fires it. replaceState rather than
    // push, so Back does not land here and open another tab, and
    // history.state carries HA's routing state across.
    setTimeout(() => {
      history.replaceState(history.state, "", home);
      window.dispatchEvent(new CustomEvent("location-changed",
        { detail: { replace: true }, bubbles: true, composed: true }));
    }, 0);
  }
}

// The tag must match panel_custom's `name:`. Guarded because define() throws
// on a name that is already registered.
if (!customElements.get("endora-console")) {
  customElements.define("endora-console", EndoraConsole);
}
