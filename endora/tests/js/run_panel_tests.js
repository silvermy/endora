// tests/js/run_panel_tests.js
//
// Executes docs/homeassistant/endora-console.js against a stub of the small
// part of the DOM it touches, and asserts what it actually does.
//
// Worth running rather than grepping: the bug that prompted it was invisible
// to inspection — window.open() returns null when passed "noopener", so a
// success check on its return value is always false.
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
  const timers = [];

  class StubElement {
    constructor() { this.innerHTML = ""; this._handlers = {}; }
    // Enough of the DOM to wire the panel's own buttons: the module looks up
    // one id and attaches a click handler to it.
    querySelector(sel) {
      if (!new RegExp(`id="${sel.replace("#", "")}"`).test(this.innerHTML)) return null;
      const el = this._handlers[sel] || (this._handlers[sel] = {
        addEventListener(type, fn) { this[type] = fn; },
      });
      return el;
    }
    click(sel) {
      const el = this._handlers[sel];
      assert.ok(el && el.click, `nothing listening on ${sel}`);
      el.click({ preventDefault() {} });
    }
  }

  const sandbox = {
    HTMLElement: StubElement,
    CustomEvent: class {
      constructor(type, init) { this.type = type; Object.assign(this, init); }
    },
    customElements: {
      _defined: {},
      get(name) { return this._defined[name]; },
      define(name, cls) { this._defined[name] = cls; },
    },
    history: {
      replaceState(...a) { calls.replaceState.push(a); },
      back() { calls.back = (calls.back || 0) + 1; },
      pushState() {
        throw new Error("pushState must not be used: it leaves the panel in history");
      },
    },
    // Navigation must be deferred past connectedCallback, so queued callbacks
    // are held until a test flushes them. A test that sees navigation before
    // flush() means the inline dispatch — which never reached HA's router —
    // has come back.
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

const HASS = { hass: { defaultPanel: "lovelace-home" } };

// The module's own TARGET, read from source so the tests do not hardcode an
// address that changes per install.
const TARGET_RE = fs.readFileSync(SRC, "utf8")
  .match(/const TARGET = "([^"]+)"/)[1]
  .replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const tests = {
  "opens the target in a new tab"() {
    const h = load({ openReturns: () => ({ opener: {} }) });
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

  "always offers a way back to Home Assistant"() {
    // The panel draws its own content with no HA toolbar, so it has no menu
    // button. In the iOS app the sidebar is hidden behind that button, which
    // left no way off this page at all — and on mobile window.open is
    // usually refused, so it takes the branch that deliberately stays put.
    for (const ret of [() => ({ opener: {} }), () => null]) {
      const h = load({ openReturns: ret });
      const el = newPanel(h, HASS);
      el.connectedCallback();
      assert.match(el.innerHTML, /id="endora-back"/, "no way back rendered");
    }
  },

  "the back button navigates home"() {
    const h = load({ openReturns: () => null });   // blocked, as on iOS
    const el = newPanel(h, HASS);
    el.connectedCallback();
    h.flush();
    assert.strictEqual(h.calls.replaceState.length, 0, "navigated unasked");
    el.click("#endora-back");
    h.flush();
    assert.strictEqual(h.calls.replaceState[0][2], "/lovelace-home");
  },

  "the back button falls back to history when no dashboard is known"() {
    const h = load({ openReturns: () => null });
    const el = newPanel(h, { hass: {} });
    el.connectedCallback();
    el.click("#endora-back");
    h.flush();
    assert.strictEqual(h.calls.replaceState.length, 0);
    assert.strictEqual(h.calls.back, 1, "no fallback out of the panel");
  },

  "renders whether or not the tab opened"() {
    for (const ret of [() => ({ opener: {} }), () => null]) {
      const h = load({ openReturns: ret });
      const el = newPanel(h, HASS);
      el.connectedCallback();
      assert.ok(el.innerHTML.trim().length > 0, "panel rendered nothing");
      assert.match(el.innerHTML, /<a [^>]*href=/, "no link out of the panel");
    }
  },

  "navigates home once the tab is open"() {
    const h = load({ openReturns: () => ({ opener: {} }) });
    newPanel(h, HASS).connectedCallback();
    h.flush();
    assert.strictEqual(h.calls.replaceState.length, 1, "did not navigate home");
    assert.strictEqual(h.calls.replaceState[0][2], "/lovelace-home");
  },

  "defers navigation past connectedCallback"() {
    const h = load({ openReturns: () => ({ opener: {} }) });
    newPanel(h, HASS).connectedCallback();
    assert.strictEqual(h.calls.replaceState.length, 0,
      "URL rewritten synchronously, before HA can act on it");
    h.flush();
    assert.strictEqual(h.calls.replaceState.length, 1);
  },

  "fires location-changed on window, where HA listens"() {
    const h = load({ openReturns: () => ({ opener: {} }) });
    newPanel(h, HASS).connectedCallback();
    h.flush();
    assert.strictEqual(h.calls.dispatched.length, 1, "no location-changed event");
    const ev = h.calls.dispatched[0];
    assert.strictEqual(ev.type, "location-changed");
    assert.ok(ev.bubbles && ev.composed, "event will not reach HA's router");
    assert.ok(ev.detail && ev.detail.replace, "HA will treat it as a push");
  },

  "preserves HA's history state when navigating"() {
    const h = load({ openReturns: () => ({ opener: {} }) });
    h.sandbox.history.state = { ha: "router-state" };
    newPanel(h, HASS).connectedCallback();
    h.flush();
    assert.deepStrictEqual(h.calls.replaceState[0][0], { ha: "router-state" });
  },

  "honours a configured home path over HA's default"() {
    const h = load({ openReturns: () => ({ opener: {} }) });
    const el = newPanel(h, { ...HASS, panel: { config: { home_path: "/dashboard-a/2" } } });
    el.connectedCallback();
    h.flush();
    assert.strictEqual(h.calls.replaceState[0][2], "/dashboard-a/2");
  },

  "does not navigate when no dashboard is known"() {
    // The bug every earlier fix missed: with hass.defaultPanel empty the
    // panel fell back to "/lovelace", which does not exist on a system whose
    // dashboards were all created by hand. HA answers such a route by
    // spinning forever, which reads as the panel having failed.
    const h = load({ openReturns: () => ({ opener: {} }) });
    const el = newPanel(h, { hass: {} });          // hass set, defaultPanel not
    el.connectedCallback();
    h.flush();
    assert.strictEqual(h.calls.replaceState.length, 0,
      "navigated to a guessed dashboard");
    assert.ok(el.innerHTML.trim().length > 0, "left the user with nothing");
  },

  "does not navigate when the popup was blocked"() {
    // Bouncing home would discard the only remaining route to the console.
    const h = load({ openReturns: () => null });
    const el = newPanel(h, HASS);
    el.connectedCallback();
    h.flush();
    assert.strictEqual(h.calls.replaceState.length, 0);
    // Assert the way out, not the wording: the copy is deliberately not
    // alarming, because on mobile a refused popup is the normal case.
    assert.match(el.innerHTML, new RegExp(`href="${TARGET_RE}"`),
      "no link to the console when the popup was refused");
  },

  "survives window.open throwing"() {
    const h = load({ openReturns: () => { throw new Error("blocked hard"); } });
    const el = newPanel(h, HASS);
    el.connectedCallback();                        // must not propagate
    assert.match(el.innerHTML, new RegExp(`href="${TARGET_RE}"`));
    assert.match(el.innerHTML, /id="endora-back"/);
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
