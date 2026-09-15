// tests/js/run_panel_tests.js
//
// Executes docs/homeassistant/endora-console.js against a stub of the small
// part of the DOM it touches, and asserts what it actually does.
//
// Worth running rather than grepping because the bug that prompted it was
// invisible to inspection: window.open() returns null when passed "noopener",
// so a success check on its return value is always false and the panel
// reports a blocked popup even when the tab opened. Only executing it shows
// that.
//
// Run standalone with `node tests/js/run_panel_tests.js`; the pytest suite
// runs it through tests/test_ha_panel_docs.py.

const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const SRC = path.join(__dirname, "..", "..", "docs", "homeassistant",
                      "endora-console.js");

/** Build a fresh sandbox, run the panel module in it, return the harness. */
function load({ openReturns }) {
  const calls = { open: [], replaceState: [], dispatched: [] };

  class StubElement {
    constructor() { this.innerHTML = ""; }
    dispatchEvent(e) { calls.dispatched.push(e); return true; }
  }

  const timers = [];
  const sandbox = {
    HTMLElement: StubElement,
    CustomEvent: class { constructor(type, init) { this.type = type; Object.assign(this, init); } },
    customElements: {
      _defined: {},
      define(name, cls) { this._defined[name] = cls; },
    },
    history: {
      replaceState(...a) { calls.replaceState.push(a); },
      pushState() { throw new Error("pushState must not be used: it leaves the panel in history"); },
    },
    // Navigation must be deferred past connectedCallback, so the harness
    // holds queued callbacks until a test explicitly flushes them. A test
    // that sees navigation before flush() would mean the inline dispatch
    // that left HA spinning on an already-changed URL had come back.
    setTimeout(fn) { timers.push(fn); },
    window: {
      open(...a) { calls.open.push(a); return openReturns(); },
      dispatchEvent(e) { calls.dispatched.push(e); return true; },
    },
  };
  sandbox.window.window = sandbox.window;

  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(SRC, "utf8"), sandbox, { filename: SRC });
  return { sandbox, calls, flush: () => { while (timers.length) timers.shift()(); } };
}

function newPanel(harness, props = {}) {
  const Cls = harness.sandbox.customElements._defined["endora-console"];
  assert.ok(Cls, "endora-console was never defined");
  const el = new Cls();
  Object.assign(el, props);
  return el;
}

const tests = {
  "opens the target in a new tab"() {
    const fake = { opener: {} };
    const h = load({ openReturns: () => fake });
    newPanel(h).connectedCallback();
    assert.strictEqual(h.calls.open.length, 1, "window.open not called");
    const [url, target] = h.calls.open[0];
    assert.match(url, /^https?:\/\/.+:\d+\/$/, `bad target url: ${url}`);
    assert.strictEqual(target, "_blank");
  },

  "does not pass noopener, which would make success undetectable"() {
    const h = load({ openReturns: () => ({ opener: {} }) });
    newPanel(h).connectedCallback();
    const features = h.calls.open[0][2];
    assert.ok(
      features === undefined || !String(features).includes("noopener"),
      `window.open passed ${features}; its return value is then always null`);
  },

  "severs the opener reference on the new window"() {
    const fake = { opener: "the HA window" };
    const h = load({ openReturns: () => fake });
    newPanel(h).connectedCallback();
    assert.strictEqual(fake.opener, null, "new tab can still reach window.opener");
  },

  "navigates Home Assistant home once the tab is open"() {
    const h = load({ openReturns: () => ({ opener: {} }) });
    newPanel(h, { hass: { defaultPanel: "lovelace-home" } }).connectedCallback();
    h.flush();
    assert.strictEqual(h.calls.replaceState.length, 1, "did not navigate home");
    assert.strictEqual(h.calls.dispatched.length, 1, "no location-changed event");
    const ev = h.calls.dispatched[0];
    assert.strictEqual(ev.type, "location-changed");
    assert.ok(ev.bubbles && ev.composed, "event will not reach HA's router");
  },

  "defers navigation past connectedCallback"() {
    // Dispatched inline, the event fired before the element was in the
    // document and HA's router never heard it — while replaceState had
    // already moved the URL, so the panel spun forever on a route that no
    // longer matched and only a reload escaped.
    const h = load({ openReturns: () => ({ opener: {} }) });
    newPanel(h, { hass: { defaultPanel: "lovelace-home" } }).connectedCallback();
    assert.strictEqual(h.calls.replaceState.length, 0,
      "URL rewritten synchronously, before HA can act on it");
    h.flush();
    assert.strictEqual(h.calls.replaceState.length, 1);
  },

  "fires location-changed on window, where HA listens"() {
    // HA's own navigate() helper fires it on window; an element-scoped
    // dispatch depends on where in the DOM the panel happens to be.
    const h = load({ openReturns: () => ({ opener: {} }) });
    const el = newPanel(h, { hass: { defaultPanel: "lovelace-home" } });
    el.connectedCallback();
    h.flush();
    assert.strictEqual(h.calls.dispatched.length, 1);
    assert.ok(h.calls.dispatched[0].detail && h.calls.dispatched[0].detail.replace,
      "detail.replace not set; HA treats it as a push");
  },

  "renders content on the success path too"() {
    // The empty panel was what made the failed navigation look like a hang:
    // nothing to read, nothing to click, and the URL already changed.
    const h = load({ openReturns: () => ({ opener: {} }) });
    const el = newPanel(h, { hass: { defaultPanel: "lovelace-home" } });
    el.connectedCallback();
    assert.ok(el.innerHTML.trim().length > 0, "success path rendered nothing");
    assert.match(el.innerHTML, /<a [^>]*href=/, "no link out of the panel");
  },

  "uses the dashboard HA says is the user's default"() {
    // The hardcoded "/lovelace" is a guess that is wrong on any install whose
    // dashboards were all created by hand — one such install had no dashboard
    // at that path at all, so the panel navigated nowhere.
    const h = load({ openReturns: () => ({ opener: {} }) });
    const el = newPanel(h, { hass: { defaultPanel: "lovelace-home" } });
    el.connectedCallback();
    h.flush();
    assert.strictEqual(h.calls.replaceState[0][2], "/lovelace-home");
  },

  "honours a configured home path over HA's default"() {
    const h = load({ openReturns: () => ({ opener: {} }) });
    const el = newPanel(h, {
      hass: { defaultPanel: "lovelace-home" },
      panel: { config: { home_path: "/dashboard-main/2" } },
    });
    el.connectedCallback();
    h.flush();
    assert.strictEqual(h.calls.replaceState[0][2], "/dashboard-main/2");
  },

  "falls back when HA offers no default panel"() {
    const h = load({ openReturns: () => ({ opener: {} }) });
    newPanel(h).connectedCallback();               // no hass property at all
    h.flush();
    assert.strictEqual(h.calls.replaceState[0][2], "/lovelace");
  },

  "stays put and offers a link when the popup is blocked"() {
    const h = load({ openReturns: () => null });
    const el = newPanel(h);
    el.connectedCallback();
    h.flush();
    assert.strictEqual(h.calls.replaceState.length, 0,
      "navigated away from the only link to the console");
    assert.match(el.innerHTML, /blocked/i);
    assert.match(el.innerHTML, /<a [^>]*target="_blank"/);
    assert.match(el.innerHTML, /rel="noopener"/);
  },

  "opens only one tab however often HA reconnects the element"() {
    const h = load({ openReturns: () => ({ opener: {} }) });
    const el = newPanel(h);
    el.connectedCallback();
    el.connectedCallback();
    el.connectedCallback();
    assert.strictEqual(h.calls.open.length, 1, "reconnect opened another tab");
  },

  "survives window.open throwing"() {
    const h = load({ openReturns: () => { throw new Error("blocked hard"); } });
    const el = newPanel(h);
    el.connectedCallback();                       // must not propagate
    assert.match(el.innerHTML, /blocked/i);
  },
};

let failed = 0;
for (const [name, fn] of Object.entries(tests)) {
  try {
    fn();
    console.log(`  PASS  ${name}`);
  } catch (e) {
    failed++;
    console.log(`  FAIL  ${name}\n        ${e.message}`);
  }
}
console.log(failed ? `${failed} failed` : `${Object.keys(tests).length} passed`);
process.exit(failed ? 1 : 0);
