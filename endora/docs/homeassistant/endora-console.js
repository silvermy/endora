// Endora — Home Assistant sidebar panel that opens the debug console in a
// new browser tab.
//
// Install
// -------
//   1. Copy this file to /config/www/endora-console.js on the Home Assistant
//      host. (/config/www/ is served at /local/; create it if missing.)
//   2. Add the panel_custom block from README.md to configuration.yaml.
//   3. Restart Home Assistant.
//
// Change TARGET below to wherever Endora's debug server is listening. It is
// logged at startup as "Debug stream: http://<host>:<port>/".
const TARGET = "http://10.0.0.142:8765/";

// Where Home Assistant should land after the console opens in its own tab.
//
// Normally nothing to set: HA tells the panel which dashboard the user has
// chosen as their default, via hass.defaultPanel. This constant is only the
// last resort if that is unavailable, and is a guess — "lovelace" is the
// stock dashboard's path, and an install whose dashboards were all created
// by hand may not have one at all. (The install this was written against
// had dashboards named lovelace-home, dashboard-cameras and so on, and no
// plain "lovelace" anywhere, so the old hardcoded default navigated to a
// dashboard that did not exist.) Override with `config: home_path:` in the
// panel_custom block.
const HOME_PATH = "/lovelace";

// Why a custom panel rather than a link or an iframe:
//
// Home Assistant hardwires sidebar entries to open in-page, so no sidebar
// item can target a new tab by configuration alone. panel_custom with
// embed_iframe false (the default) loads this module directly into HA's
// frontend instead of framing it, which means the code runs on HA's own page
// and can call window.open itself.
//
// An iframe would not work regardless: HA is served over HTTPS and the
// console over HTTP, and mixed-content blocking rejects an HTTP iframe inside
// an HTTPS page. It does not reject top-level navigation, which is why
// opening a tab and the fallback link below are both fine.
class EndoraConsole extends HTMLElement {
  connectedCallback() {
    // HA can connect and disconnect a panel element more than once; without
    // this guard, returning to the panel opens another tab each time.
    if (this._ran) return;
    this._ran = true;

    // The sidebar click is still a live user activation at this point, so
    // the popup blocker normally allows this. "Normally" is not "always" —
    // it is a timing-dependent allowance, not a guarantee — so a blocked
    // popup has to stay on screen with a link rather than leave a blank
    // panel.
    //
    // "noopener" is deliberately NOT passed here: window.open() returns null
    // when it is, success or not, which makes the outcome undetectable and
    // the message below a coin flip. Clearing .opener on the new window
    // achieves the same isolation while leaving a usable return value.
    let opened = null;
    try {
      opened = window.open(TARGET, "_blank");
      if (opened) {
        try { opened.opener = null; } catch (e) { /* cross-origin; harmless */ }
      }
    } catch (e) {
      opened = null;
    }

    if (opened) {
      this._goHome();
      return;
    }

    this.innerHTML = `
      <div style="padding:24px;font-family:var(--paper-font-body1_-_font-family,sans-serif);
                  color:var(--primary-text-color,#212121)">
        <p>Your browser blocked the new tab.</p>
        <p><a href="${TARGET}" target="_blank" rel="noopener"
              style="color:var(--primary-color,#03a9f4)">
          Open the Endora debug console &#8599;</a></p>
      </div>`;
  }

  // Send Home Assistant back to the dashboard once the console is open, so
  // it never rests on this panel. Without it, every later reload that
  // restores this route opens another tab — which is what makes allowing
  // pop-ups for HA safe rather than annoying.
  //
  // Only on success: bouncing home after a blocked popup would discard the
  // fallback link and leave no way to reach the console at all.
  _goHome() {
    const cfg = (this.panel && this.panel.config) || {};
    // Explicit config wins; otherwise ask HA which dashboard this user set as
    // their default, and only guess if it will not say.
    const preferred = this.hass && this.hass.defaultPanel;
    const home = cfg.home_path || (preferred ? "/" + preferred : HOME_PATH);
    // replaceState, not pushState: this panel must not stay in history, or
    // the browser's Back button returns to it and opens a further tab.
    history.replaceState(null, "", home);
    this.dispatchEvent(
      new CustomEvent("location-changed", { bubbles: true, composed: true }));
  }
}

// The element tag must match the panel_custom `name:` option, and — being a
// custom element — must contain a hyphen.
customElements.define("endora-console", EndoraConsole);
