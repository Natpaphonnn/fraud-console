// Threshold simulator UI. All numbers come from the page data (#site-data), which
// scripts/build_web.py builds from reports/*.json. Logic lives in sim.js.
import {
  PARAM_KEYS,
  bestIndex,
  costCurve,
  metricsAt,
  nearestIndex,
  validateParams,
} from "./sim.js";

const DATA = JSON.parse(document.getElementById("site-data").textContent);
const D = DATA.sim;
const SVG_NS = "http://www.w3.org/2000/svg";
const X_MIN = 1e-4; // left edge of the log threshold axis (threshold 0 is drawn at the edge)

const $ = (id) => document.getElementById(id);
const fmtPct = (x, nd = 1) => (Number.isFinite(x) ? `${(100 * x).toFixed(nd)}%` : "—");
const fmtUsd = (x) => `${Math.round(x).toLocaleString("en-US")} USD`;
const fmtNum = (x, nd = 1) => x.toLocaleString("en-US", { maximumFractionDigits: nd, minimumFractionDigits: nd });
const fmtThr = (t) => (t === 0 ? "0" : t < 0.001 ? t.toPrecision(2) : t.toFixed(4));

const inputs = {
  review_cost_per_alert: $("c-review"),
  fp_friction_cost: $("c-friction"),
  fn_loss_fraction: $("c-fnfrac"),
  fn_fixed_cost: $("c-fnfixed"),
};
const slider = $("thr");

const state = {
  index: nearestIndex(D, D.chosen_threshold),
  params: { ...D.default_params },
  curve: null,
  best: 0,
};

function readParams() {
  const p = {};
  for (const k of PARAM_KEYS) {
    const raw = inputs[k].value.trim();
    p[k] = raw === "" ? NaN : Number(raw);
  }
  return p;
}

function writeParams(p) {
  for (const k of PARAM_KEYS) inputs[k].value = String(p[k]);
}

function recomputeCurve() {
  state.curve = costCurve(D, state.params);
  state.best = bestIndex(D, state.params);
}

function render() {
  const m = metricsAt(D, state.index, state.params);
  const best = metricsAt(D, state.best, state.params);
  slider.value = String(state.index);
  slider.setAttribute("aria-valuetext", `threshold ${fmtThr(m.threshold)}`);
  $("thr-out").textContent = fmtThr(m.threshold);
  $("o-recall").textContent = fmtPct(m.recall);
  $("o-precision").textContent = fmtPct(m.precision);
  $("o-alert").textContent = fmtPct(m.alertRate, 2);
  $("o-cost").textContent = fmtUsd(m.costPer1000);
  $("o-cost-sub").textContent =
    `vs ${fmtUsd(m.nothingPer1000)} stepping up nobody` +
    (Number.isFinite(m.saving) ? ` (${m.saving >= 0 ? `${fmtPct(m.saving)} saved` : `${fmtPct(-m.saving)} more`})` : "");
  $("o-sentence").textContent =
    `Per 1,000 transactions: ${fmtNum(m.flaggedPer1000)} stepped up, of which ${fmtNum(m.fraudCaughtPer1000)} ` +
    `are fraud; ${fmtNum(m.fraudMissedPer1000)} fraud missed (${fmtUsd(m.missedUsdPer1000)} of fraud amount).`;
  $("o-best").textContent =
    `Lowest cost for these costs: threshold ${fmtThr(best.threshold)} → ${fmtUsd(best.costPer1000)} per 1,000, ` +
    `recall ${fmtPct(best.recall)}.`;
  drawChart(m);
}

// ---------- chart ----------

function el(name, attrs = {}, parent) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (parent) parent.appendChild(node);
  return node;
}

function niceStep(range, target) {
  const raw = range / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const r = raw / mag;
  return (r <= 1 ? 1 : r <= 2 ? 2 : r <= 2.5 ? 2.5 : r <= 5 ? 5 : 10) * mag;
}

const box = $("cost-chart");
const tooltip = document.createElement("div");
tooltip.className = "tooltip";
tooltip.hidden = true;
box.appendChild(tooltip);
let geom = null;

function drawChart(current) {
  const W = Math.max(280, box.clientWidth);
  const H = Math.round(Math.min(340, Math.max(220, W * 0.5)));
  const M = { top: 14, right: 14, bottom: 40, left: 64 };
  const per1000 = 1000 / D.n_rows;
  const ys = state.curve.map((c) => c * per1000);
  const nothing = current.nothingPer1000;
  const yTop = Math.max(nothing * 1.35, ys[state.best] * 1.6, 1);
  const step = niceStep(yTop, 4);
  const yMax = Math.ceil(yTop / step) * step;
  const lx0 = Math.log10(X_MIN);
  const xOf = (t) => M.left + ((Math.log10(Math.max(t, X_MIN)) - lx0) / -lx0) * (W - M.left - M.right);
  const yOf = (v) => M.top + (1 - Math.min(v, yMax * 1.05) / yMax) * (H - M.top - M.bottom);

  const svg = el("svg", {
    viewBox: `0 0 ${W} ${H}`,
    width: W,
    height: H,
    role: "img",
    "aria-label":
      `Expected cost per 1,000 transactions against the threshold. Current threshold ${fmtThr(current.threshold)} ` +
      `costs ${fmtUsd(current.costPer1000)}; stepping up nobody costs ${fmtUsd(nothing)}.`,
  });
  const clipId = "plot-clip";
  const defs = el("defs", {}, svg);
  const clip = el("clipPath", { id: clipId }, defs);
  el("rect", { x: M.left, y: M.top, width: W - M.left - M.right, height: H - M.top - M.bottom }, clip);

  const grid = el("g", { class: "grid" }, svg);
  const axes = el("g", { class: "axis" }, svg);
  for (let v = 0; v <= yMax + 1e-9; v += step) {
    const y = yOf(v);
    el("line", { x1: M.left, x2: W - M.right, y1: y, y2: y }, grid);
    const t = el("text", { x: M.left - 8, y: y + 4, "text-anchor": "end" }, axes);
    t.textContent = Math.round(v).toLocaleString("en-US");
  }
  for (let e = -4; e <= 0; e++) {
    const x = xOf(10 ** e);
    el("line", { x1: x, x2: x, y1: M.top, y2: H - M.bottom }, grid);
    const t = el("text", { x, y: H - M.bottom + 18, "text-anchor": "middle" }, axes);
    t.textContent = e === 0 ? "1" : (10 ** e).toString();
  }
  const xl = el("text", { x: (M.left + W - M.right) / 2, y: H - 4, "text-anchor": "middle", class: "axis-title" }, axes);
  xl.textContent = "Step-up threshold (model score)";
  const yl = el("text", { x: 12, y: M.top + (H - M.top - M.bottom) / 2, "text-anchor": "middle", class: "axis-title",
    transform: `rotate(-90 12 ${M.top + (H - M.top - M.bottom) / 2})` }, axes);
  yl.textContent = "USD per 1,000";

  const plot = el("g", { "clip-path": `url(#${clipId})` }, svg);
  el("line", { class: "ref", x1: M.left, x2: W - M.right, y1: yOf(nothing), y2: yOf(nothing) }, plot);
  let dPath = "";
  D.threshold.forEach((t, i) => {
    if (t <= 0) return;
    dPath += `${dPath ? "L" : "M"}${xOf(t).toFixed(1)},${yOf(ys[i]).toFixed(1)}`;
  });
  el("path", { class: "curve", d: dPath }, plot);

  const cx = xOf(current.threshold);
  el("line", { class: "now-line", x1: cx, x2: cx, y1: M.top, y2: H - M.bottom }, svg);
  const bx = xOf(D.threshold[state.best]);
  el("circle", { class: "best-dot", cx: bx, cy: yOf(ys[state.best]), r: 6 }, svg);
  el("circle", { class: "now-dot", cx, cy: yOf(current.costPer1000), r: 5 }, svg);
  const hover = el("line", { class: "hover-line", x1: 0, x2: 0, y1: M.top, y2: H - M.bottom, visibility: "hidden" }, svg);

  box.querySelector("svg")?.remove();
  box.insertBefore(svg, tooltip);
  geom = { W, H, M, lx0, hover, svg };
}

function indexAtPointer(evt) {
  const r = geom.svg.getBoundingClientRect();
  const x = ((evt.clientX - r.left) / r.width) * geom.W;
  const frac = (x - geom.M.left) / (geom.W - geom.M.left - geom.M.right);
  if (frac < -0.05 || frac > 1.05) return null;
  const t = 10 ** (geom.lx0 * (1 - Math.min(1, Math.max(0, frac))));
  return nearestIndex(D, t);
}

box.addEventListener("pointermove", (evt) => {
  if (!geom) return;
  const i = indexAtPointer(evt);
  if (i === null) return hideTip();
  const m = metricsAt(D, i, state.params);
  const r = geom.svg.getBoundingClientRect();
  const sx = r.width / geom.W;
  const lx0 = geom.lx0;
  const x = geom.M.left + ((Math.log10(Math.max(m.threshold, X_MIN)) - lx0) / -lx0) * (geom.W - geom.M.left - geom.M.right);
  geom.hover.setAttribute("x1", x);
  geom.hover.setAttribute("x2", x);
  geom.hover.setAttribute("visibility", "visible");
  tooltip.innerHTML = "";
  const lines = [
    ["Threshold", fmtThr(m.threshold)],
    ["Cost / 1,000", fmtUsd(m.costPer1000)],
    ["Recall", fmtPct(m.recall)],
    ["Precision", fmtPct(m.precision)],
  ];
  for (const [k, v] of lines) {
    const row = document.createElement("div");
    const a = document.createElement("span");
    a.textContent = k;
    const b = document.createElement("strong");
    b.textContent = v;
    row.append(a, b);
    tooltip.appendChild(row);
  }
  tooltip.hidden = false;
  const left = x * sx;
  const tw = tooltip.offsetWidth;
  tooltip.style.left = `${Math.min(Math.max(4, left + 12), r.width - tw - 4)}px`;
  tooltip.style.top = `${geom.M.top * sx + 4}px`;
});
box.addEventListener("pointerleave", hideTip);
box.addEventListener("click", (evt) => {
  if (!geom) return;
  const i = indexAtPointer(evt);
  if (i === null) return;
  state.index = i;
  render();
});

function hideTip() {
  tooltip.hidden = true;
  if (geom) geom.hover.setAttribute("visibility", "hidden");
}

// ---------- wiring ----------

function onCostInput() {
  const p = readParams();
  const errors = validateParams(p);
  const box = $("cost-error");
  for (const k of PARAM_KEYS) inputs[k].setAttribute("aria-invalid", String(errors.some((e) => e.startsWith(k))));
  if (errors.length) {
    box.textContent = "Please enter non-negative numbers (the share of the amount lost must be between 0 and 2).";
    box.hidden = false;
    return;
  }
  box.hidden = true;
  state.params = p;
  recomputeCurve();
  render();
}

function init() {
  slider.min = "0";
  slider.max = String(D.threshold.length - 1);
  for (const node of document.querySelectorAll("#sim-form [disabled]")) node.disabled = false;
  slider.addEventListener("input", () => {
    state.index = Number(slider.value);
    render();
  });
  for (const k of PARAM_KEYS) inputs[k].addEventListener("input", onCostInput);
  $("btn-best").addEventListener("click", () => {
    state.index = state.best;
    render();
  });
  $("btn-project").addEventListener("click", () => {
    state.index = nearestIndex(D, D.chosen_threshold);
    render();
  });
  $("sim-form").addEventListener("reset", (evt) => {
    evt.preventDefault();
    writeParams(D.default_params);
    state.index = nearestIndex(D, D.chosen_threshold);
    onCostInput();
  });
  $("sim-form").addEventListener("submit", (evt) => evt.preventDefault());
  let pending = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(render);
  }).observe(box);
  recomputeCurve();
  render();
}

init();
