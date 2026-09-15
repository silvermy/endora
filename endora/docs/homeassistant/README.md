# Opening the debug console from the Home Assistant sidebar

The debug console is served over plain HTTP on the machine running Endora,
while Home Assistant is usually served over HTTPS. That rules out the two
obvious approaches and leaves one that works.

## Why not an iframe

`panel_iframe`, a Lovelace **Webpage** card, and the add-on's own ingress
panel all embed the target page. Mixed-content blocking rejects an HTTP
iframe inside an HTTPS page, so the panel comes up blank with a console
error and no visible explanation.

Top-level *navigation* from HTTPS to HTTP is not blocked — only embedded
subresources are. So the answer is to open the console as a page, not to
embed it.

## Why not just a sidebar link

Home Assistant hardwires sidebar entries to open in-page. There is no
configuration option that makes one target a new tab.

## What works

`panel_custom` with `embed_iframe: false` (the default) loads a JavaScript
module directly into HA's frontend rather than framing it. The module runs on
HA's own page, so it can call `window.open` itself.

### 1. Install the module

Copy [`endora-console.js`](endora-console.js) to `/config/www/` on the Home
Assistant host, creating that directory if it does not exist:

```
/config/www/endora-console.js
```

Edit the `TARGET` constant at the top of the file to point at your Endora
host. The address is logged at startup:

```
Debug stream: http://10.0.0.142:8765/
```

### 2. Add the panel

In `configuration.yaml`:

```yaml
panel_custom:
  - name: endora-console
    sidebar_title: Endora
    sidebar_icon: mdi:hand-wave
    url_path: endora-console
    module_url: /local/endora-console.js
    require_admin: true
```

`name` is the custom element tag and must match the `customElements.define`
call in the JS, hyphen included. `require_admin` is deliberate: the console
can change detection settings and restart behaviour.

Once the console opens, the panel sends Home Assistant back to the dashboard
so it never rests on this route — otherwise every later reload that restores
it opens another tab.

It goes to whichever dashboard HA reports as your default, so there is
usually nothing to configure. To send it somewhere specific instead:

```yaml
    config:
      home_path: /lovelace-home
```

Use a real `url_path` from **Settings → Dashboards**. Do not assume
`/lovelace` exists: on an install whose dashboards were all created by hand
there may be no dashboard at that path at all.

### 3. Restart Home Assistant

## Notes

- **Two sidebar entries.** Running Endora as an add-on already contributes an
  "Endora" ingress panel. Hide either one by long-pressing the sidebar header
  and dragging it into the hidden section — that is per-user and changes
  nothing else.
- **`/local/` is cached hard.** After editing the JS, bump the URL
  (`module_url: /local/endora-console.js?v=2`) or the browser will keep
  serving the old copy.
- **If the tab does not open**, the browser's popup blocker stopped
  `window.open`. The panel detects this, stays where it is and renders a link
  instead, so it is one click rather than a failure. To let it through, allow
  pop-ups for your HA origin: in Chrome and Edge, click the blocked-popup
  icon at the right of the address bar and choose "Always allow"; in Safari,
  Settings → Websites → Pop-up Windows; in Firefox, the notification bar's
  Options button. That is per browser, so repeat it on a phone or another
  machine.
