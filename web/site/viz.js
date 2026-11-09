// Hand-written Canvas/SVG visuals for the console: decision flow (Sankey + moving dots),
// score strip, convergence chart, PR curve, number tweening. No libraries.
// Every visual reads the app state through `ctx` (scores, labels, rows played, threshold),
// the same state the counters use; computations live in flow.js.
import { convergence, dotVisible, laneCounts, outcomeOf, prCurve } from "./flow.js";
import { currentLang, t } from "./i18n.js";

export const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
export const X_MIN = 1e-4;
const SVG_NS = "http://www.w3.org/2000/svg";
const LX0 = Math.log10(X_MIN);

export const logPos = (s) => Math.min(1, Math.max(0, (Math.log10(Math.max(s, X_MIN)) - LX0) / -LX0));
export const fromLogPos = (f) => 10 ** (LX0 * (1 - Math.min(1, Math.max(0, f))));

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
export function palette() {
  return {
    accent: cssVar("--accent"),
    danger: cssVar("--danger"),
    ink: cssVar("--ink"),
    ink2: cssVar("--ink-2"),
    ink3: cssVar("--ink-3"),
    line: cssVar("--line"),
    line2: cssVar("--line-2"),
    bg2: cssVar("--bg-2"),
  };
}

// deterministic 0..1 per row (jitter that does not change between frames)
function hash01(i) {
  let x = Math.imul(i + 1, 0x9e3779b1) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x85ebca6b) >>> 0;
  x ^= x >>> 13;
  return (x >>> 0) / 4294967296;
}

function setupCanvas(canvas, w, h) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return g;
}

function svg(name, attrs = {}, parent) {
  const n = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (parent) parent.appendChild(n);
  return n;
}

const lerp = (a, b, s) => a + (b - a) * s;
const smooth = (s) => s * s * (3 - 2 * s);

// ------------------------------------------------------------------ number tween (≤ 300 ms)

export function tweenNumber(el, value, fmt, ms = 260) {
  el.dataset.value = Number.isFinite(value) ? String(value) : "";
  const from = el._v;
  el._v = value;
  cancelAnimationFrame(el._raf);
  if (REDUCED || !Number.isFinite(value) || !Number.isFinite(from) || from === value) {
    el.textContent = fmt(value);
    return;
  }
  const t0 = performance.now();
  const step = (now) => {
    const s = Math.min(1, (now - t0) / ms);
    const e = 1 - (1 - s) ** 3;
    el.textContent = fmt(s >= 1 ? value : from + (value - from) * e);
    if (s < 1) el._raf = requestAnimationFrame(step);
  };
  el._raf = requestAnimationFrame(step);
}

// ------------------------------------------------------------------ decision flow

const OUTCOMES = [
  ["tp", "●"],
  ["fp", "○"],
  ["fn", "×"],
  ["tn", "·"],
];

export function createFlow(root, ctx) {
  const box = root.querySelector(".flow-box");
  const canvas = box.querySelector("canvas");
  const make = (cls, html) => {
    const el = document.createElement("div");
    el.className = cls;
    if (html) el.innerHTML = html;
    box.append(el);
    return el;
  };
  const lbl = {
    source: make("fl fl-source", `<span data-i18n="js.flow.stream">${t("js.flow.stream")}</span>`),
    model: make(
      "fl fl-model",
      `<span class="k" data-i18n="js.flow.model">${t("js.flow.model")}</span><b class="fl-score">—</b>`,
    ),
    step: make("fl fl-lane fl-step", '<span>STEP-UP</span> <b data-k="stepup">0</b>'),
    approve: make("fl fl-lane fl-approve", '<span>APPROVE</span> <b data-k="approve">0</b>'),
  };
  const buckets = {};
  for (const [k, glyph] of OUTCOMES) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `fl fl-bucket fl-${k}`;
    b.dataset.outcome = k;
    b.innerHTML =
      `<span class="g">${glyph}</span> ` +
      `<span class="nm" data-i18n="js.flow.${k}">${t(`js.flow.${k}`)}</span> <b>0</b>`;
    b.addEventListener("click", () => ctx.selectLatest(k));
    box.append(b);
    buckets[k] = b;
  }
  const scoreEl = lbl.model.querySelector(".fl-score");

  let W = 0;
  let H = 0;
  let vertical = false;
  let g = null;
  let colors = palette();
  let dots = [];
  let positions = [];
  let lastKey = "";
  let view = "all";
  const pad = { l: 12, r: 12, t: 14, b: 14 };

  const L = () =>
    vertical
      ? { laneV: { step: 0.27, approve: 0.73 }, outV: { tp: 0.12, fp: 0.38, fn: 0.62, tn: 0.88 }, S: 0.3 }
      : { laneV: { step: 0.25, approve: 0.69 }, outV: { tp: 0.1, fp: 0.36, fn: 0.55, tn: 0.84 }, S: 0.26 };
  const U = { model0: 0.29, model1: 0.39, lane0: 0.41, lane1: 0.51, lane2: 0.66, out1: 0.78, end: 0.88 };
  const cross = () => (vertical ? W - pad.l - pad.r : H - pad.t - pad.b);
  const P = (u, v) =>
    vertical
      ? [pad.l + v * (W - pad.l - pad.r), pad.t + u * (H - pad.t - pad.b)]
      : [pad.l + u * (W - pad.l - pad.r), pad.t + v * (H - pad.t - pad.b)];

  function place(el, u, v, dx = 0, dy = 0) {
    const [x, y] = P(u, v);
    el.style.left = `${x + dx}px`;
    el.style.top = `${y + dy}px`;
  }

  function resize() {
    W = box.clientWidth;
    vertical = W < 640;
    H = vertical ? 540 : 280;
    box.style.height = `${H}px`;
    box.classList.toggle("vertical", vertical);
    g = setupCanvas(canvas, W, H);
    const l = L();
    place(lbl.source, 0, 0.5);
    place(lbl.model, (U.model0 + U.model1) / 2, 0.5);
    place(lbl.step, (U.lane1 + U.lane2) / 2, l.laneV.step);
    place(lbl.approve, (U.lane1 + U.lane2) / 2, l.laneV.approve);
    for (const [k] of OUTCOMES) place(buckets[k], U.end, l.outV[k]);
    lastKey = "";
  }

  function thick(count, n) {
    if (!count || !n) return 0;
    return Math.max(ctx.minLanePx, (cross() * L().S * count) / n);
  }

  // band from (u0,v0) straight to u1, S-curve to (u2,v1), straight to u3; width in px
  function band(u0, v0, u1, u2, v1, u3, width, color) {
    if (width <= 0) return;
    const um = (u1 + u2) / 2;
    g.beginPath();
    g.moveTo(...P(u0, v0));
    g.lineTo(...P(u1, v0));
    g.bezierCurveTo(...P(um, v0), ...P(um, v1), ...P(u2, v1));
    g.lineTo(...P(u3, v1));
    g.lineWidth = width;
    g.strokeStyle = color;
    g.stroke();
  }

  function drawBands(c) {
    const l = L();
    const n = c.n;
    const cr = cross();
    g.globalAlpha = 1;
    g.lineCap = "butt";
    if (!n) {
      // empty: show the structure as thin guides
      g.setLineDash([3, 4]);
      for (const [lane, outs] of [["step", ["tp", "fp"]], ["approve", ["fn", "tn"]]]) {
        band(0, 0.5, U.lane0, U.lane1, l.laneV[lane], U.lane2, 1, colors.line2);
        for (const o of outs) band(U.lane2, l.laneV[lane], U.lane2, U.out1, l.outV[o], U.end, 1, colors.line2);
      }
      g.setLineDash([]);
    } else {
      const tN = thick(n, n);
      const tS = thick(c.stepup, n);
      const tA = thick(c.approve, n);
      const t = { tp: thick(c.tp, n), fp: thick(c.fp, n), fn: thick(c.fn, n), tn: thick(c.tn, n) };
      const alpha = (hex, a) => {
        g.globalAlpha = a;
        return hex;
      };
      band(0, 0.5, U.lane0, U.lane0, 0.5, U.lane0, tN, alpha(colors.ink3, 0.28));
      const vs = 0.5 + (-tN / 2 + tS / 2) / cr;
      const va = 0.5 + (-tN / 2 + tS + tA / 2) / cr;
      band(U.lane0, vs, U.lane0, U.lane1, l.laneV.step, U.lane2, tS, alpha(colors.accent, 0.42));
      band(U.lane0, va, U.lane0, U.lane1, l.laneV.approve, U.lane2, tA, alpha(colors.ink3, 0.28));
      const sTop = l.laneV.step - tS / 2 / cr;
      const aTop = l.laneV.approve - tA / 2 / cr;
      band(U.lane2, sTop + t.tp / 2 / cr, U.lane2, U.out1, l.outV.tp, U.end, t.tp, alpha(colors.accent, 0.75));
      band(U.lane2, sTop + (t.tp + t.fp / 2) / cr, U.lane2, U.out1, l.outV.fp, U.end, t.fp, alpha(colors.ink2, 0.35));
      band(U.lane2, aTop + t.fn / 2 / cr, U.lane2, U.out1, l.outV.fn, U.end, t.fn, alpha(colors.danger, 0.75));
      band(U.lane2, aTop + (t.fn + t.tn / 2) / cr, U.lane2, U.out1, l.outV.tn, U.end, t.tn, alpha(colors.ink3, 0.22));
      g.globalAlpha = 1;
    }
    // model box
    const [x0, y0] = P(U.model0, 0.5 - L().S / 2 - 0.06);
    const [x1, y1] = P(U.model1, 0.5 + L().S / 2 + 0.06);
    const rx = Math.min(x0, x1);
    const ry = Math.min(y0, y1);
    g.fillStyle = colors.bg2;
    g.strokeStyle = colors.line2;
    g.lineWidth = 1;
    g.beginPath();
    g.roundRect(rx, ry, Math.abs(x1 - x0), Math.abs(y1 - y0), 5);
    g.fill();
    g.stroke();
  }

  function dotPos(d, now, t, c) {
    const D = (2600 / Math.sqrt(ctx.speed())) | 0;
    const p = (now - d.t0) / D;
    if (p < 0 || p >= 1) return null;
    const l = L();
    const s = ctx.scores[d.i];
    const step = s >= t;
    const lane = step ? "step" : "approve";
    const o = outcomeOf(s, ctx.y[d.i], t);
    const cr = cross();
    const j = hash01(d.i) - 0.5;
    const laneW = (step ? thick(c.stepup, c.n) : thick(c.approve, c.n)) / cr;
    const outW = thick(c[o], c.n) / cr;
    const jl = j * Math.min(laneW * 0.8, 10 / cr);
    const jo = j * Math.min(outW * 0.8, 10 / cr);
    let u;
    let v;
    let phase = "pre";
    if (p < 0.3) [u, v] = [lerp(0, U.model0 + 0.02, p / 0.3), 0.5 + j * (8 / cr)];
    else if (p < 0.4) [u, v, phase] = [(U.model0 + U.model1) / 2, 0.5, "model"];
    else if (p < 0.55) {
      const s2 = smooth((p - 0.4) / 0.15);
      [u, v] = [lerp(U.lane0, U.lane1, (p - 0.4) / 0.15), lerp(0.5, l.laneV[lane] + jl, s2)];
    } else if (p < 0.68) [u, v] = [lerp(U.lane1, U.lane2, (p - 0.55) / 0.13), l.laneV[lane] + jl];
    else if (p < 0.82) {
      const s2 = smooth((p - 0.68) / 0.14);
      [u, v, phase] = [lerp(U.lane2, U.out1, (p - 0.68) / 0.14), lerp(l.laneV[lane] + jl, l.outV[o] + jo, s2), "out"];
    } else [u, v, phase] = [lerp(U.out1, U.end, (p - 0.82) / 0.18), l.outV[o] + jo, "out"];
    const [x, y] = P(u, v);
    return { x, y, p, step, o, phase, i: d.i };
  }

  function drawDot(q) {
    const a = q.p > 0.9 ? (1 - q.p) / 0.1 : 1;
    g.globalAlpha = Math.max(0, a);
    if (q.phase !== "out") {
      g.fillStyle = q.step ? colors.accent : colors.ink3;
      g.beginPath();
      g.arc(q.x, q.y, q.step ? 3.2 : 2.6, 0, Math.PI * 2);
      g.fill();
    } else if (q.o === "tp") {
      g.fillStyle = colors.accent;
      g.beginPath();
      g.arc(q.x, q.y, 3.8, 0, Math.PI * 2);
      g.fill();
    } else if (q.o === "fp") {
      g.strokeStyle = colors.ink2;
      g.lineWidth = 1.4;
      g.beginPath();
      g.arc(q.x, q.y, 3.2, 0, Math.PI * 2);
      g.stroke();
    } else if (q.o === "fn") {
      g.strokeStyle = colors.danger;
      g.lineWidth = 1.8;
      g.beginPath();
      g.moveTo(q.x - 3.4, q.y - 3.4);
      g.lineTo(q.x + 3.4, q.y + 3.4);
      g.moveTo(q.x + 3.4, q.y - 3.4);
      g.lineTo(q.x - 3.4, q.y + 3.4);
      g.stroke();
    } else {
      g.fillStyle = colors.ink3;
      g.beginPath();
      g.arc(q.x, q.y, 1.9, 0, Math.PI * 2);
      g.fill();
    }
    g.globalAlpha = 1;
  }

  function updateLabels(c) {
    root.querySelector('[data-k="stepup"]').textContent = c.stepup.toLocaleString("en-US");
    root.querySelector('[data-k="approve"]').textContent = c.approve.toLocaleString("en-US");
    for (const [k] of OUTCOMES) {
      const b = buckets[k];
      const name = t(`js.flow.${k}`);
      b.dataset.count = String(c[k]);
      b.querySelector("b").textContent = c[k].toLocaleString("en-US");
      b.title = t("js.flow.bucket_title", { name });
      b.setAttribute("aria-label", t("js.flow.bucket_aria", { name, count: c[k], n: c.n }));
      b.disabled = c[k] === 0;
    }
    lbl.step.dataset.count = String(c.stepup);
    lbl.approve.dataset.count = String(c.approve);
    root.dataset.n = String(c.n);
  }

  function frame(now) {
    if (!g) return;
    const t = ctx.thr();
    const c = laneCounts(ctx.scores, ctx.y, ctx.played(), t);
    const key = `${c.tp},${c.fp},${c.fn},${c.tn},${W},${H},${t},${colors.accent}`;
    if (!dots.length && key === lastKey) return;
    if (key !== lastKey) updateLabels(c);
    lastKey = key;
    g.clearRect(0, 0, W, H);
    drawBands(c);
    positions = [];
    let inModel = null;
    const alive = [];
    for (const d of dots) {
      const q = dotPos(d, now, t, c);
      if (!q) {
        if (now - d.t0 < 0) alive.push(d); // not started yet
        continue;
      }
      alive.push(d);
      positions.push(q);
      if (q.phase === "model" && (!inModel || q.i > inModel.i)) inModel = q;
      drawDot(q);
    }
    dots = alive;
    if (inModel) {
      scoreEl.textContent = ctx.fmtScore(ctx.scores[inModel.i]);
      scoreEl.classList.toggle("step", inModel.step);
    }
    canvas.dataset.animating = String(dots.length > 0);
  }

  canvas.addEventListener("click", (e) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    let best = null;
    let bd = 14 * 14;
    for (const q of positions) {
      const d2 = (q.x - x) ** 2 + (q.y - y) ** 2;
      if (d2 < bd) [best, bd] = [q, d2];
    }
    if (best) ctx.select(best.i);
  });

  new ResizeObserver(() => resize()).observe(box);
  resize();
  root.dataset.static = String(REDUCED);

  return {
    spawn(i, delay = 0) {
      if (REDUCED || !dotVisible(i, ctx.y[i], view, ctx.dotEvery)) return;
      dots.push({ i, t0: performance.now() + delay });
    },
    clear() {
      dots = [];
      lastKey = "";
      scoreEl.textContent = "—";
    },
    setView(v) {
      view = v;
      dots = dots.filter((d) => dotVisible(d.i, ctx.y[d.i], view, ctx.dotEvery));
    },
    refresh() {
      colors = palette();
      lastKey = "";
    },
    invalidate() {
      lastKey = "";
    },
    frame,
  };
}

// ------------------------------------------------------------------ score strip

export function createStrip(root, ctx) {
  const canvas = root.querySelector("canvas");
  const handle = root.querySelector(".strip-handle");
  let W = 0;
  const H = 58;
  let g = null;
  let colors = palette();
  let lastKey = "";

  function resize() {
    W = root.querySelector(".strip-plot").clientWidth;
    g = setupCanvas(canvas, W, H);
    lastKey = "";
  }

  function render() {
    if (!g) return;
    const n = ctx.played();
    const thr = ctx.thr();
    const key = `${n},${thr},${W},${colors.accent},${currentLang()}`;
    if (key === lastKey) return;
    lastKey = key;
    g.clearRect(0, 0, W, H);
    g.fillStyle = colors.line;
    g.fillRect(0, H / 2, W, 1);
    // legit first, fraud on top
    for (const fraud of [0, 1]) {
      g.fillStyle = fraud ? colors.accent : colors.ink3;
      g.globalAlpha = fraud ? 1 : 0.55;
      const r = fraud ? 3.2 : 1.7;
      for (let i = 0; i < n; i++) {
        if (ctx.y[i] !== fraud) continue;
        const x = logPos(ctx.scores[i]) * (W - 8) + 4;
        const y = 6 + hash01(i) * (H - 12);
        g.beginPath();
        g.arc(x, y, r, 0, Math.PI * 2);
        g.fill();
      }
    }
    g.globalAlpha = 1;
    const hx = logPos(thr) * (W - 8) + 4;
    handle.style.left = `${hx}px`;
    handle.setAttribute("aria-valuetext", t("js.thr.valuetext", { t: ctx.fmtScore(thr) }));
    handle.setAttribute("aria-valuenow", String(ctx.index()));
    root.dataset.dots = String(n);
  }

  const plot = root.querySelector(".strip-plot");
  let dragging = false;
  const toThreshold = (e) => {
    const r = plot.getBoundingClientRect();
    ctx.setThreshold(fromLogPos((e.clientX - r.left - 4) / (r.width - 8)));
  };
  plot.addEventListener("pointerdown", (e) => {
    dragging = true;
    plot.setPointerCapture(e.pointerId);
    toThreshold(e);
  });
  plot.addEventListener("pointermove", (e) => dragging && toThreshold(e));
  plot.addEventListener("pointerup", () => (dragging = false));
  plot.addEventListener("pointercancel", () => (dragging = false));
  handle.addEventListener("keydown", (e) => {
    const step = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -10, PageUp: 10 }[e.key];
    if (step) {
      e.preventDefault();
      ctx.setIndex(ctx.index() + step);
    }
  });

  new ResizeObserver(() => {
    resize();
    render();
  }).observe(plot);
  resize();
  return {
    render,
    refresh() {
      colors = palette();
      lastKey = "";
    },
  };
}

// ------------------------------------------------------------------ convergence chart

export function createConvergence(root, ctx) {
  const box = root.querySelector(".conv-box");
  let colors = palette();
  let lastKey = "";

  function render() {
    const n = ctx.played();
    const thr = ctx.thr();
    const W = Math.max(240, box.clientWidth);
    const key = `${n},${thr},${W},${colors.accent},${currentLang()}`;
    if (key === lastKey) return;
    lastKey = key;
    const H = 132;
    const M = { l: 34, r: 10, t: 8, b: 20 };
    const c = convergence(ctx.scores, ctx.y, n, thr);
    const full = ctx.full();
    const x = (j) => M.l + ((j + 1) / Math.max(n, 1)) * (W - M.l - M.r);
    const y = (v) => M.t + (1 - v) * (H - M.t - M.b);
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" });
    s.setAttribute(
      "aria-label",
      t("js.conv.aria", { n, ci: "95%" }),
    );
    const axis = svg("g", { class: "axis" }, s);
    for (const v of [0, 0.5, 1]) {
      svg("line", { x1: M.l, x2: W - M.r, y1: y(v), y2: y(v), class: "gridline" }, axis);
      svg("text", { x: M.l - 6, y: y(v) + 3.5, "text-anchor": "end" }, axis).textContent = `${v * 100}%`;
    }
    svg("text", { x: M.l, y: H - 5 }, axis).textContent = t("js.conv.x");
    svg("text", { x: W - M.r, y: H - 5, "text-anchor": "end" }, axis).textContent = n.toLocaleString("en-US");
    const stride = Math.max(1, Math.ceil(n / (W - M.l - M.r)));
    const idx = [];
    for (let j = 0; j < n; j += stride) idx.push(j);
    if (n && idx.at(-1) !== n - 1) idx.push(n - 1);
    const series = [
      ["recall", c.recall, c.recallLo, c.recallHi, full.recall],
      ["precision", c.precision, c.precLo, c.precHi, full.precision],
    ];
    for (const [name, v, lo, hi, ref] of series) {
      const ok = idx.filter((j) => Number.isFinite(v[j]));
      if (Number.isFinite(ref)) {
        svg("line", { x1: M.l, x2: W - M.r, y1: y(ref), y2: y(ref), class: `ref ref--${name}` }, s);
      }
      if (ok.length) {
        const top = ok.map((j) => `${x(j).toFixed(1)},${y(hi[j]).toFixed(1)}`);
        const bot = ok.map((j) => `${x(j).toFixed(1)},${y(lo[j]).toFixed(1)}`).reverse();
        svg("polygon", { points: [...top, ...bot].join(" "), class: `band band--${name}` }, s);
        const d = ok.map((j, k) => `${k ? "L" : "M"}${x(j).toFixed(1)},${y(v[j]).toFixed(1)}`).join("");
        svg("path", { d, class: `line line--${name}` }, s);
      }
    }
    box.replaceChildren(s);
    root.dataset.n = String(n);
    root.dataset.recallLast = n ? String(c.recall.at(-1)) : "";
    root.dataset.precisionLast = n ? String(c.precision.at(-1)) : "";
    root.dataset.recallFull = String(full.recall);
    root.dataset.precisionFull = String(full.precision);
  }

  new ResizeObserver(() => {
    lastKey = "";
    render();
  }).observe(box);
  return {
    render,
    refresh() {
      colors = palette();
      lastKey = "";
    },
    invalidate() {
      lastKey = "";
    },
  };
}

// ------------------------------------------------------------------ PR curve

export function createPR(root, ctx, d) {
  const box = root.querySelector(".pr-box");
  const pts = prCurve(d);
  const noSkill = d.n_fraud / d.n_rows;
  let lastKey = "";
  let geom = null;

  function render() {
    const W = Math.max(240, box.clientWidth);
    const i = ctx.index();
    const key = `${W},${i},${currentLang()}`;
    if (key === lastKey) return;
    lastKey = key;
    const H = Math.round(Math.min(300, Math.max(210, W * 0.5)));
    const M = { l: 44, r: 12, t: 10, b: 36 };
    const x = (r) => M.l + r * (W - M.l - M.r);
    const y = (p) => M.t + (1 - p) * (H - M.t - M.b);
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" });
    const axis = svg("g", { class: "axis" }, s);
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      svg("line", { x1: M.l, x2: W - M.r, y1: y(v), y2: y(v), class: "gridline" }, axis);
      svg("text", { x: M.l - 6, y: y(v) + 3.5, "text-anchor": "end" }, axis).textContent = `${v * 100}%`;
      svg("text", { x: x(v), y: H - M.b + 15, "text-anchor": "middle" }, axis).textContent = `${v * 100}%`;
    }
    svg("text", { x: (M.l + W - M.r) / 2, y: H - 4, "text-anchor": "middle", class: "axis-title" }, axis).textContent =
      t("js.pr.x");
    const yc = M.t + (H - M.t - M.b) / 2;
    svg("text", { x: 11, y: yc, "text-anchor": "middle", class: "axis-title", transform: `rotate(-90 11 ${yc})` }, axis).textContent =
      t("js.pr.y");
    svg("line", { x1: M.l, x2: W - M.r, y1: y(noSkill), y2: y(noSkill), class: "ref" }, s);
    svg("text", { x: W - M.r, y: y(noSkill) - 5, "text-anchor": "end", class: "note" }, s).textContent =
      t("js.pr.noskill", { x: noSkill.toFixed(4) });
    const dd = pts.map((p, k) => `${k ? "L" : "M"}${x(p.recall).toFixed(1)},${y(p.precision).toFixed(1)}`).join("");
    svg("path", { d: dd, class: "pr-line", pathLength: 1 }, s);
    const cur = pts.find((p) => p.i === i) ?? pts.reduce((a, b) => (Math.abs(b.i - i) < Math.abs(a.i - i) ? b : a));
    const g = svg("g", { class: "pr-now" }, s);
    svg("circle", { cx: x(cur.recall), cy: y(cur.precision), r: 5, class: "pr-dot" }, g);
    const lx = x(cur.recall) > W * 0.6 ? x(cur.recall) - 10 : x(cur.recall) + 10;
    svg("text", { x: lx, y: y(cur.precision) - 9, "text-anchor": lx < x(cur.recall) ? "end" : "start", class: "note strong" }, g).textContent =
      t("js.pr.now", { t: ctx.fmtScore(d.threshold[cur.i]) });
    root.dataset.recall = String(cur.recall);
    root.dataset.precision = String(cur.precision);
    root.dataset.index = String(cur.i);
    s.setAttribute(
      "aria-label",
      t("js.pr.aria", {
        t: ctx.fmtScore(d.threshold[cur.i]),
        r: `${(100 * cur.recall).toFixed(1)}%`,
        p: `${(100 * cur.precision).toFixed(1)}%`,
      }),
    );
    box.replaceChildren(s);
    geom = { x, y, W, H, M };
  }

  box.addEventListener("click", (e) => {
    if (!geom) return;
    const r = box.querySelector("svg").getBoundingClientRect();
    const sx = geom.W / r.width;
    const px = (e.clientX - r.left) * sx;
    const py = (e.clientY - r.top) * sx;
    let best = null;
    let bd = Infinity;
    for (const p of pts) {
      const d2 = (geom.x(p.recall) - px) ** 2 + (geom.y(p.precision) - py) ** 2;
      if (d2 < bd) [best, bd] = [p, d2];
    }
    if (best && bd < 30 * 30) ctx.setIndex(best.i);
  });

  // draw the curve the first time it scrolls into view (CSS transition on stroke-dashoffset)
  if (REDUCED) root.classList.add("revealed");
  else {
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((en) => en.isIntersecting)) {
          io.disconnect();
          requestAnimationFrame(() => root.classList.add("revealed"));
        }
      },
      { threshold: 0.35 },
    );
    io.observe(root);
  }

  new ResizeObserver(() => {
    lastKey = "";
    render();
  }).observe(box);
  return { render };
}
