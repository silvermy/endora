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
    if (this._opened) return;
    this._opened = true;

    // The sidebar click is still a live user activation at this point, so
    // the popup blocker normally allows this. "Normally" is not "always" —
    // it is a timing-dependent allowance, not a guarantee — so the panel
    // always renders the link too and says which happened. A blocked popup
    // then costs one click rather than leaving a blank panel.
    let opened = null;
    try {
      opened = window.open(TARGET, "_blank", "noopener");
    } catch (e) {
      opened = null;
    }

    this.innerHTML = `
      <div style="padding:24px;font-family:var(--paper-font-body1_-_font-family,sans-serif);
                  color:var(--primary-text-color,#212121)">
        <p>${opened
              ? "Opened the Endora debug console in a new tab."
              : "Your browser blocked the new tab."}</p>
        <p><a href="${TARGET}" target="_blank" rel="noopener"
              style="color:var(--primary-color,#03a9f4)">
          Open the Endora debug console &#8599;</a></p>
      </div>`;
  }
}

// The element tag must match the panel_custom `name:` option, and — being a
// custom element — must contain a hyphen.
customElements.define("endora-console", EndoraConsole);
