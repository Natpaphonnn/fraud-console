// Fraud console. All numbers come from the page data (#site-data), built by
// scripts/build_web.py from reports/. Scoring logic: model.js; cost logic: sim.js.
import { outcomeOf } from "./flow.js";
import { contributions, prepareModel, score, streamRow } from "./model.js";
import {
  PARAM_KEYS,
  bestIndex,
  costCurve,
  metricsAt,
  nearestIndex,
  sampleOutcomes,
  validateParams,
} from "./sim.js";
import { createConvergence, createFlow, createPR, createStrip, tweenNumber } from "./viz.js";
import { currentLang, initialLang, setLang, t } from "./i18n.js";

const DATA = JSON.parse(document.getElementById("site-data").textContent);
const D = DATA.sim;
const S = DATA.stream;
const FOLD = Object.fromEntries(Object.entries(DATA.model.fold_models).map(([k, m]) => [k, prepareModel(m)]));
const FINAL = prepareModel(DATA.model.final_model);
const N = S.id.length;
const ROWS = Array.from({ length: N }, (_, i) => streamRow(S, i));
const SCORES = ROWS.map((r) => score(FOLD[String(r.fold)], r)); // out-of-fold, as in CV
const Y = ROWS.map((r) => r.y);
const AMT = ROWS.map((r) => r.amount_usd);
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const VISIBLE = 14;
const ROWS_PER_SEC = 2;
const X_MIN = 1e-4;
const SVG_NS = "http://www.w3.org/2000/svg";
const PER = "1,000"; // costs are shown per 1,000 transactions

const $ = (id) => document.getElementById(id);
const fmtPct = (x, nd = 1) => (Number.isFinite(x) ? `${(100 * x).toFixed(nd)}%` : "—");
const fmtUsd = (x) => `${Math.round(x).toLocaleString("en-US")} USD`;
const fmtNum = (x, nd = 1) => x.toLocaleString("en-US", { maximumFractionDigits: nd, minimumFractionDigits: nd });
// One format for every score and threshold on the page: 4 decimals, "<0.0001" below that.
const fmt4 = (v) => (v === 0 ? "0" : v < 0.0001 ? "<0.0001" : v.toFixed(4));
const fmtThr = fmt4;
const fmtScore = fmt4;
const fmtAmt = (x) => `$${x.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const fmtSigned = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(2)}`;
const logPos = (s) => Math.min(1, Math.max(0, (Math.log10(Math.max(s, X_MIN)) - Math.log10(X_MIN)) / -Math.log10(X_MIN)));
const h = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const state = {
  index: nearestIndex(D, D.chosen_threshold),
  params: { ...D.default_params },
  curve: null,
  best: 0,
  played: 0,
  playing: !REDUCED,
  speed: 1,
  selected: null, // stream row index shown in the inspector
  pinned: false,
  pg: null, // playground transaction
};
const thr = () => D.threshold[state.index];

// ---------------------------------------------------------------- shared threshold

function setIndex(i) {
  state.index = Math.max(0, Math.min(D.threshold.length - 1, i));
  renderSim();
  renderStream();
  renderCounters();
  renderInspector();
  renderPlayground();
  viz.flow?.invalidate();
  viz.pr?.render();
  markDirty();
}

// visuals that redraw at most once per animation frame
const viz = {};
let dirty = true;
const markDirty = () => {
  dirty = true;
};

function selectRow(i) {
  state.pinned = true;
  state.selected = i;
  renderStream();
  renderInspector();
}

// ---------------------------------------------------------------- live stream

const GLYPH = { tp: "●", fn: "×", fp: "○", tn: "·" };
function outcome(i) {
  const o = outcomeOf(SCORES[i], Y[i], thr());
  return [o, GLYPH[o], t(`js.out.${o}`)];
}

function sbar(s, thrv) {
  const wrap = h("span", "sbar");
  const track = h("span", "sbar-track");
  track.title = t("js.sbar_title", { score: fmtScore(s), min: X_MIN, max: 1, t: fmtThr(thrv) });
  const fill = h("span", `sbar-fill${s >= thrv ? " step" : ""}`);
  fill.style.width = `${(100 * logPos(s)).toFixed(1)}%`;
  const mark = h("span", "sbar-thr");
  mark.style.left = `${(100 * logPos(thrv)).toFixed(1)}%`;
  track.append(fill, mark);
  wrap.append(track, h("span", "sbar-val", fmtScore(s)));
  return wrap;
}

// rows with index >= freshFrom are new this tick: a caught fraud flashes once
function renderStream(freshFrom = Infinity) {
  const list = $("s-rows");
  const tv = thr();
  const frag = document.createDocumentFragment();
  for (let k = state.played - 1; k >= Math.max(0, state.played - VISIBLE); k--) {
    const r = ROWS[k];
    const [cls, glyph, label] = outcome(k);
    const li = h("li");
    if (k >= freshFrom && cls === "tp" && !REDUCED) li.className = "flash-tp";
    const b = h("button", "srow");
    b.type = "button";
    b.dataset.i = String(k);
    b.setAttribute("aria-pressed", String(state.selected === k));
    b.setAttribute(
      "aria-label",
      t("js.row_aria", {
        id: r.id,
        amount: fmtAmt(r.amount_usd),
        score: fmtScore(SCORES[k]),
        decision: t(SCORES[k] >= tv ? "js.dec.step" : "js.dec.approve"),
        outcome: label,
      }),
    );
    const out = h("span", `out ${cls}`);
    out.append(h("span", `o o--${cls}`, glyph), h("span", "txt", label));
    b.append(
      h("span", "id", `#${String(r.id).padStart(5, "0")}`),
      h("span", "amt", fmtAmt(r.amount_usd)),
      h("span", "merch", r.merchant_category),
      h("span", "chan", r.channel),
      sbar(SCORES[k], tv),
      h("span", `dec${SCORES[k] >= tv ? " step" : ""}`, SCORES[k] >= tv ? "STEP-UP" : "APPROVE"),
      out,
    );
    li.append(b);
    frag.append(li);
  }
  list.replaceChildren(frag);
  $("s-progress").textContent = t("js.progress", { n: state.played.toLocaleString("en-US") });
}

function renderCounters() {
  const tv = thr();
  const full = sampleOutcomes(SCORES, Y, AMT, tv, state.params);
  const pct = (nd) => (v) => fmtPct(v, nd);
  const int = (v) => Math.round(v).toLocaleString("en-US");
  tweenNumber($("f-recall"), full.recall, pct(1));
  tweenNumber($("f-precision"), full.precision, pct(1));
  tweenNumber($("f-alert"), full.alertRate, pct(2));
  tweenNumber($("f-cost"), full.costPer1000, fmtUsd);
  $("f-counts").textContent = t("js.f_counts", {
    tp: full.tp,
    fn: full.fn,
    fp: full.fp.toLocaleString("en-US"),
    tn: full.tn.toLocaleString("en-US"),
  });
  const k = state.played;
  const so = sampleOutcomes(SCORES.slice(0, k), Y.slice(0, k), AMT.slice(0, k), tv, state.params);
  tweenNumber($("c-tp"), so.tp, int);
  tweenNumber($("c-fn"), so.fn, int);
  tweenNumber($("c-fp"), so.fp, int);
  tweenNumber($("c-tn"), so.tn, int);
  tweenNumber($("c-recall"), so.recall, pct(0));
  tweenNumber($("c-precision"), so.precision, pct(0));
  for (const [id, v] of [["c-tp", so.tp], ["c-fn", so.fn], ["c-fp", so.fp], ["c-tn", so.tn]]) $(id).dataset.count = String(v);
  $("c-played").textContent = t("js.c_played", { k: k.toLocaleString("en-US"), n: N.toLocaleString("en-US") });
}

function renderPlayState() {
  const btn = $("s-play");
  btn.textContent = t(state.playing ? "js.play.pause" : state.played >= N ? "js.play.replay" : "js.play.play");
  btn.setAttribute("aria-pressed", String(state.playing));
  $("live-dot").classList.toggle("on", state.playing);
  const hint = $("s-hint");
  if (!state.playing && state.played === 0 && REDUCED) {
    hint.textContent = t("js.hint.reduced");
    hint.hidden = false;
  } else if (state.played >= N) {
    hint.textContent = t("js.hint.end");
    hint.hidden = false;
  } else hint.hidden = true;
}

function advance(n) {
  const before = state.played;
  state.played = Math.min(N, state.played + n);
  if (state.played === before) return;
  if (!state.pinned) state.selected = state.played - 1;
  const gap = 1000 / (ROWS_PER_SEC * state.speed);
  for (let i = before; i < state.played; i++) viz.flow?.spawn(i, (i - before) * gap);
  renderStream(before);
  renderCounters();
  renderInspector();
  markDirty();
  if (state.played >= N) {
    state.playing = false;
    renderPlayState();
  }
}

let acc = 0;
let last = 0;
function loop(ts) {
  if (state.playing) {
    acc += (Math.min(250, ts - (last || ts)) / 1000) * ROWS_PER_SEC * state.speed;
    if (acc >= 1) {
      const n = Math.floor(acc);
      acc -= n;
      advance(n);
    }
  }
  last = ts;
  viz.flow?.frame(ts);
  if (dirty) {
    dirty = false;
    viz.strip?.render();
    viz.conv?.render();
  }
  requestAnimationFrame(loop);
}

function togglePlay() {
  if (state.played >= N) restart();
  state.playing = !state.playing;
  renderPlayState();
}

function step() {
  state.playing = false;
  advance(1);
  renderPlayState();
}

function restart() {
  state.played = 0;
  state.selected = null;
  state.pinned = false;
  acc = 0;
  viz.flow?.clear();
  renderStream();
  renderCounters();
  renderInspector();
  renderPlayState();
  markDirty();
}

function setSpeed(v) {
  state.speed = v;
  for (const b of document.querySelectorAll(".seg [data-speed]")) {
    b.setAttribute("aria-checked", String(Number(b.dataset.speed) === v));
  }
}

// ---------------------------------------------------------------- contributions

const NUM_DP = { amount_usd: 2, account_balance_usd: 2 };
function fmtValue(f, v) {
  if (DATA.model.final_model.boolean.some((b) => b.feature === f)) return t(Number(v) ? "js.yes" : "js.no");
  if (typeof v === "string") return v;
  const dp = NUM_DP[f] ?? (Number.isInteger(v) ? 0 : 2);
  return v.toLocaleString("en-US", { maximumFractionDigits: dp, minimumFractionDigits: dp });
}

function contribList(pm, row, top = 7, grow = false) {
  const items = contributions(pm, row).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const shown = items.slice(0, top);
  const rest = items.slice(top);
  const restSum = rest.reduce((s, [, c]) => s + c, 0);
  const logit = pm.intercept + items.reduce((s, [, c]) => s + c, 0);
  const scale = Math.max(1, ...shown.map(([, c]) => Math.abs(c)), Math.abs(restSum));
  const ul = h("ul", `contrib${grow && !REDUCED ? " grow" : ""}`);
  ul.setAttribute("aria-label", t("js.contrib.aria"));
  const line = (label, valueLabel, c, bar = true, cls = "") => {
    const li = h("li", cls);
    const f = h("span", "f");
    f.append(document.createTextNode(label));
    if (valueLabel !== null) f.append(h("i", null, `= ${valueLabel}`));
    const track = h("span", "t");
    if (bar) {
      const b = h("span", `b${c >= 0 ? " pos" : ""}`);
      const w = (50 * Math.abs(c)) / scale;
      b.style.width = `${w.toFixed(2)}%`;
      b.style.left = c >= 0 ? "50%" : `${(50 - w).toFixed(2)}%`;
      track.append(b);
    }
    li.append(f, track, h("span", "n", fmtSigned(c)));
    return li;
  };
  for (const [feat, c] of shown) ul.append(line(feat, fmtValue(feat, row[feat]), c));
  if (rest.length) ul.append(line(t("js.contrib.others", { n: rest.length }), null, restSum));
  ul.append(line(t("js.contrib.intercept"), null, pm.intercept, false));
  ul.append(line(t("js.contrib.logodds"), null, logit, false, "sum"));
  return ul;
}

// ---------------------------------------------------------------- inspector

let lastInspected = null;
function renderInspector() {
  const body = $("insp-body");
  const mode = $("insp-mode");
  mode.textContent = t(state.pinned ? "js.insp.pinned" : "js.insp.following");
  mode.classList.toggle("pinned", state.pinned);
  if (state.selected === null) {
    body.replaceChildren(h("p", "fine", t("js.insp.waiting")));
    return;
  }
  const i = state.selected;
  const r = ROWS[i];
  const s = SCORES[i];
  const isStep = s >= thr();
  const head = h("div", "insp-head");
  head.append(
    h("span", null, `#${String(r.id).padStart(5, "0")} · ${fmtAmt(r.amount_usd)} · ${r.merchant_category} · ${r.channel}`),
    h("span", null, t("js.insp.truth", { truth: t(r.y ? "js.truth.fraud" : "js.truth.legit") })),
    h("span", "insp-score", fmtScore(s)),
    h("span", `decision${isStep ? " step" : ""}`, isStep ? "STEP-UP" : "APPROVE"),
  );
  const note = h("p", "fine", t("js.insp.note", { fold: r.fold }));
  const actions = h("div", "insp-actions");
  const toPg = h("button", "btn btn--ghost", t("js.insp.edit"));
  toPg.type = "button";
  toPg.addEventListener("click", () => {
    loadPlayground(ROWS[i]);
    goTo("playground");
  });
  actions.append(toPg);
  if (state.pinned) {
    const follow = h("button", "btn btn--ghost", t("js.insp.follow"));
    follow.type = "button";
    follow.addEventListener("click", () => {
      state.pinned = false;
      state.selected = state.played ? state.played - 1 : null;
      renderStream();
      renderInspector();
    });
    actions.append(follow);
  }
  const grow = lastInspected !== i;
  lastInspected = i;
  body.replaceChildren(head, contribList(FOLD[String(r.fold)], r, 7, grow), note, actions);
}

// ---------------------------------------------------------------- playground

const GROUPS = [
  ["transaction", ["amount_usd", "merchant_category", "channel", "time_of_day_hour", "day_of_week", "is_foreign_transaction", "is_new_merchant"]],
  ["card", ["card_type", "auth_method", "card_age_months", "customer_age", "account_balance_usd", "prior_disputes"]],
  ["risk", ["merchant_risk_score", "velocity_score", "cvv_retry_count", "txn_count_last_24h", "hours_since_last_txn", "distance_from_home_km"]],
  ["device", ["device_type", "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]],
];
const CATS = Object.fromEntries(FINAL.cat.map((b) => [b.feature, b.categories]));
const BOOLS = new Set(FINAL.bool.map((b) => b.feature));

function buildPlaygroundForm() {
  const listed = GROUPS.flatMap(([, fs]) => fs);
  const missing = DATA.model.features.filter((f) => !listed.includes(f));
  const groups = missing.length ? [...GROUPS, ["other", missing]] : GROUPS;
  const root = $("pg-fields");
  for (const [name, feats] of groups) {
    const fs = h("fieldset", "pg-group");
    const lg = h("legend", "k", t(`js.pg.group.${name}`));
    lg.dataset.i18n = `js.pg.group.${name}`; // relabelled by applyStatic() on a language switch
    const grid = h("div", "pg-grid");
    for (const f of feats) {
      if (!DATA.model.features.includes(f)) continue;
      let field;
      if (BOOLS.has(f)) {
        field = h("label", "pg-field check");
        const cb = h("input");
        cb.type = "checkbox";
        cb.dataset.f = f;
        field.append(cb, h("span", null, f));
      } else {
        field = h("label", "pg-field");
        field.append(h("span", null, f));
        let input;
        if (CATS[f]) {
          input = h("select");
          for (const c of CATS[f]) input.append(new Option(c, c));
        } else {
          const vals = S.cols[f];
          input = h("input");
          input.type = "number";
          input.min = String(Math.min(...vals));
          input.max = String(Math.max(...vals));
          input.step = vals.every(Number.isInteger) ? "1" : "any";
          input.inputMode = "decimal";
          input.setAttribute("aria-describedby", "pg-range-help");
        }
        input.dataset.f = f;
        field.append(input);
      }
      grid.append(field);
    }
    fs.append(lg, grid);
    root.append(fs);
  }
  const help = h("p", "fine", t("js.pg.help"));
  help.id = "pg-range-help";
  help.dataset.i18n = "js.pg.help";
  root.append(help);
  $("pg-form").addEventListener("input", onPlaygroundInput);
  $("pg-form").addEventListener("change", onPlaygroundInput);
  $("pg-form").addEventListener("submit", (e) => e.preventDefault());
}

function loadPlayground(row) {
  state.pg = {};
  for (const f of DATA.model.features) state.pg[f] = row[f];
  for (const el of $("pg-fields").querySelectorAll("[data-f]")) {
    const v = state.pg[el.dataset.f];
    if (el.type === "checkbox") el.checked = Boolean(Number(v));
    else el.value = String(v);
    el.removeAttribute("aria-invalid");
  }
  state.pgSource = { id: row.id, y: row.y };
  renderPlayground();
}

function onPlaygroundInput(e) {
  const el = e.target;
  const f = el.dataset?.f;
  if (!f) return;
  if (el.type === "checkbox") state.pg[f] = el.checked ? 1 : 0;
  else if (CATS[f]) state.pg[f] = el.value;
  else {
    const v = el.value.trim() === "" ? NaN : Number(el.value);
    const ok = Number.isFinite(v) && v >= 0 && !(f === "amount_usd" && v <= 0);
    el.setAttribute("aria-invalid", String(!ok));
    if (!ok) return;
    state.pg[f] = v;
  }
  state.pgSource = null; // edited by hand
  renderPlayground();
}

function renderPlayground() {
  if (!state.pg) return;
  const s = score(FINAL, state.pg);
  const tv = thr();
  const isStep = s >= tv;
  $("pg-score").textContent = fmtScore(s);
  const dec = $("pg-decision");
  dec.textContent = isStep ? "STEP-UP" : "APPROVE";
  dec.className = `decision${isStep ? " step" : ""}`;
  const g = $("pg-gauge");
  g.classList.toggle("step", isStep);
  g.querySelector(".gauge-dot").style.left = `${(100 * logPos(s)).toFixed(2)}%`;
  g.querySelector(".gauge-thr").style.left = `${(100 * logPos(tv)).toFixed(2)}%`;
  const src = state.pgSource
    ? t("js.pg.src_row", {
        id: String(state.pgSource.id).padStart(5, "0"),
        truth: t(state.pgSource.y ? "js.truth.fraud" : "js.truth.legit"),
      })
    : t("js.pg.src_edit");
  $("pg-note").textContent = t("js.pg.note", { t: fmtThr(tv), min: X_MIN, max: 1, src });
  $("pg-contrib").replaceChildren(contribList(FINAL, state.pg, 8));
}

function preset(kind) {
  if (kind === "inspector") {
    if (state.selected !== null) loadPlayground(ROWS[state.selected]);
    return;
  }
  const pool = ROWS.filter((r) => r.y === (kind === "fraud" ? 1 : 0));
  loadPlayground(pool[Math.floor(Math.random() * pool.length)]);
}

// ---------------------------------------------------------------- threshold simulator

const inputs = {
  review_cost_per_alert: $("c-review"),
  fp_friction_cost: $("c-friction"),
  fn_loss_fraction: $("c-fnfrac"),
  fn_fixed_cost: $("c-fnfixed"),
};

function recomputeCurve() {
  state.curve = costCurve(D, state.params);
  state.best = bestIndex(D, state.params);
}

function renderSim() {
  const m = metricsAt(D, state.index, state.params);
  const best = metricsAt(D, state.best, state.params);
  for (const id of ["thr", "thr-mini"]) {
    const s = $(id);
    s.value = String(state.index);
    s.setAttribute("aria-valuetext", t("js.thr.valuetext", { t: fmtThr(m.threshold) }));
  }
  $("thr-out").textContent = fmtThr(m.threshold);
  $("thr-mini-out").textContent = fmtThr(m.threshold);
  $("o-recall").textContent = fmtPct(m.recall);
  $("o-precision").textContent = fmtPct(m.precision);
  $("o-alert").textContent = fmtPct(m.alertRate, 2);
  $("o-cost").textContent = fmtUsd(m.costPer1000);
  const change = Number.isFinite(m.saving)
    ? t(m.saving >= 0 ? "js.sim.saved" : "js.sim.more", { pct: fmtPct(Math.abs(m.saving)) })
    : "—";
  $("o-cost-sub").textContent = t("js.sim.cost_sub", { nothing: fmtUsd(m.nothingPer1000), change });
  $("o-sentence").textContent = t("js.sim.sentence", {
    per: PER,
    flagged: fmtNum(m.flaggedPer1000),
    caught: fmtNum(m.fraudCaughtPer1000),
    missed: fmtNum(m.fraudMissedPer1000),
    usd: fmtUsd(m.missedUsdPer1000),
  });
  $("o-best").textContent = t("js.sim.best", {
    t: fmtThr(best.threshold),
    cost: fmtUsd(best.costPer1000),
    per: PER,
    recall: fmtPct(best.recall),
  });
  drawChart(m);
}

function onCostInput() {
  const p = {};
  for (const k of PARAM_KEYS) {
    const raw = inputs[k].value.trim();
    p[k] = raw === "" ? NaN : Number(raw);
  }
  const errors = validateParams(p);
  for (const k of PARAM_KEYS) inputs[k].setAttribute("aria-invalid", String(errors.some((e) => e.startsWith(k))));
  const box = $("cost-error");
  if (errors.length) {
    box.textContent = t("js.sim.error", { max: DATA.ui.fn_frac_max });
    box.hidden = false;
    return;
  }
  box.hidden = true;
  state.params = p;
  recomputeCurve();
  renderSim();
  renderCounters();
}

function resetCosts() {
  for (const k of PARAM_KEYS) inputs[k].value = String(D.default_params[k]);
  onCostInput();
  setIndex(nearestIndex(D, D.chosen_threshold));
}

// chart
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
const tooltip = h("div", "tooltip");
tooltip.hidden = true;
box.append(tooltip);
let geom = null;

function drawChart(current) {
  const W = Math.max(280, box.clientWidth);
  const H = Math.round(Math.min(320, Math.max(210, W * 0.46)));
  const M = { top: 12, right: 12, bottom: 38, left: 58 };
  const per1000 = 1000 / D.n_rows;
  const ys = state.curve.map((c) => c * per1000);
  const nothing = current.nothingPer1000;
  const yTop = Math.max(nothing * 1.35, ys[state.best] * 1.6, 1);
  const stepY = niceStep(yTop, 4);
  const yMax = Math.ceil(yTop / stepY) * stepY;
  const lx0 = Math.log10(X_MIN);
  const xOf = (t) => M.left + ((Math.log10(Math.max(t, X_MIN)) - lx0) / -lx0) * (W - M.left - M.right);
  const yOf = (v) => M.top + (1 - Math.min(v, yMax * 1.05) / yMax) * (H - M.top - M.bottom);
  const svg = el("svg", {
    viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
    "aria-label": t("js.chart.aria", {
      per: PER,
      t: fmtThr(current.threshold),
      cost: fmtUsd(current.costPer1000),
      nothing: fmtUsd(nothing),
    }),
  });
  const clip = el("clipPath", { id: "plot-clip" }, el("defs", {}, svg));
  el("rect", { x: M.left, y: M.top, width: W - M.left - M.right, height: H - M.top - M.bottom }, clip);
  const grid = el("g", { class: "grid" }, svg);
  const axes = el("g", { class: "axis" }, svg);
  for (let v = 0; v <= yMax + 1e-9; v += stepY) {
    el("line", { x1: M.left, x2: W - M.right, y1: yOf(v), y2: yOf(v) }, grid);
    el("text", { x: M.left - 8, y: yOf(v) + 4, "text-anchor": "end" }, axes).textContent = Math.round(v).toLocaleString("en-US");
  }
  for (let e = -4; e <= 0; e++) {
    const x = xOf(10 ** e);
    el("line", { x1: x, x2: x, y1: M.top, y2: H - M.bottom }, grid);
    el("text", { x, y: H - M.bottom + 16, "text-anchor": "middle" }, axes).textContent = e === 0 ? "1" : (10 ** e).toString();
  }
  el("text", { x: (M.left + W - M.right) / 2, y: H - 4, "text-anchor": "middle", class: "axis-title" }, axes).textContent = t("js.chart.x");
  const yc = M.top + (H - M.top - M.bottom) / 2;
  el("text", { x: 11, y: yc, "text-anchor": "middle", class: "axis-title", transform: `rotate(-90 11 ${yc})` }, axes).textContent = t("js.chart.y", { per: PER });
  const plot = el("g", { "clip-path": "url(#plot-clip)" }, svg);
  el("line", { class: "ref", x1: M.left, x2: W - M.right, y1: yOf(nothing), y2: yOf(nothing) }, plot);
  let d = "";
  D.threshold.forEach((t, i) => {
    if (t > 0) d += `${d ? "L" : "M"}${xOf(t).toFixed(1)},${yOf(ys[i]).toFixed(1)}`;
  });
  el("path", { class: "curve", d }, plot);
  const cx = xOf(current.threshold);
  el("line", { class: "now-line", x1: cx, x2: cx, y1: M.top, y2: H - M.bottom }, svg);
  el("circle", { class: "best-dot", cx: xOf(D.threshold[state.best]), cy: yOf(ys[state.best]), r: 5.5 }, svg);
  el("circle", { class: "now-dot", cx, cy: yOf(current.costPer1000), r: 4.5 }, svg);
  const hover = el("line", { class: "hover-line", x1: 0, x2: 0, y1: M.top, y2: H - M.bottom, visibility: "hidden" }, svg);
  box.querySelector("svg")?.remove();
  box.insertBefore(svg, tooltip);
  geom = { W, M, lx0, hover, svg, xOf };
}

function indexAtPointer(evt) {
  const r = geom.svg.getBoundingClientRect();
  const x = ((evt.clientX - r.left) / r.width) * geom.W;
  const frac = (x - geom.M.left) / (geom.W - geom.M.left - geom.M.right);
  if (frac < -0.05 || frac > 1.05) return null;
  return nearestIndex(D, 10 ** (geom.lx0 * (1 - Math.min(1, Math.max(0, frac)))));
}

function hideTip() {
  tooltip.hidden = true;
  if (geom) geom.hover.setAttribute("visibility", "hidden");
}

box.addEventListener("pointermove", (evt) => {
  if (!geom) return;
  const i = indexAtPointer(evt);
  if (i === null) return hideTip();
  const m = metricsAt(D, i, state.params);
  const r = geom.svg.getBoundingClientRect();
  const x = geom.xOf(m.threshold);
  geom.hover.setAttribute("x1", x);
  geom.hover.setAttribute("x2", x);
  geom.hover.setAttribute("visibility", "visible");
  tooltip.replaceChildren(
    ...[
      [t("js.tip.threshold"), fmtThr(m.threshold)],
      [t("js.tip.cost", { per: PER }), fmtUsd(m.costPer1000)],
      [t("js.tip.recall"), fmtPct(m.recall)],
      [t("js.tip.precision"), fmtPct(m.precision)],
    ].map(
      ([k, v]) => {
        const row = h("div");
        row.append(h("span", null, k), h("strong", null, v));
        return row;
      },
    ),
  );
  tooltip.hidden = false;
  const sx = r.width / geom.W;
  tooltip.style.left = `${Math.min(Math.max(4, x * sx + 12), r.width - tooltip.offsetWidth - 4)}px`;
  tooltip.style.top = `${geom.M.top * sx + 4}px`;
});
box.addEventListener("pointerleave", hideTip);
box.addEventListener("click", (evt) => {
  if (!geom) return;
  const i = indexAtPointer(evt);
  if (i !== null) setIndex(i);
});

// ---------------------------------------------------------------- navigation, theme, palette

function goTo(id) {
  const target = $(id);
  if (!target) return;
  if (target.tagName === "DETAILS") target.open = true;
  target.closest("details")?.setAttribute("open", "");
  target.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" });
  const focusable = target.matches("details") ? target.querySelector("summary") : target;
  if (!focusable.hasAttribute("tabindex") && focusable.tagName !== "SUMMARY") focusable.setAttribute("tabindex", "-1");
  focusable.focus({ preventScroll: true });
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("fe2e-theme", theme);
  } catch {
    /* storage blocked */
  }
  syncThemeButton();
  if (state.curve) renderSim();
  for (const v of Object.values(viz)) v.refresh?.();
  viz.pr?.render();
  markDirty();
}
function syncThemeButton() {
  const light = document.documentElement.dataset.theme === "light";
  const b = $("theme-toggle");
  b.setAttribute("aria-pressed", String(light));
  b.querySelector(".sr-only").textContent = t(light ? "js.theme.dark" : "js.theme.light");
  document.querySelector('meta[name="theme-color"]').content = light ? "#f5f4f0" : "#0a0c0f";
}
const toggleTheme = () => setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");

// ---------------------------------------------------------------- language

// Re-render everything the script writes; static text was swapped by setLang() (data-i18n).
function rerenderTexts() {
  renderSim();
  renderStream();
  renderCounters();
  renderInspector();
  renderPlayState();
  renderPlayground();
  syncThemeButton();
  for (const v of Object.values(viz)) v.invalidate?.();
  viz.pr?.render();
  markDirty();
  if (pal.open) renderPalette();
}

async function switchLang(next) {
  try {
    await setLang(next);
  } catch (err) {
    console.warn("language switch failed", err);
    return;
  }
  rerenderTexts();
}
const toggleLang = () => switchLang(currentLang() === "en" ? "th" : "en");

// [group key, label key, action]; labels are looked up when the palette renders (current language)
const COMMANDS = [
  ["section", "flow", () => goTo("flow")],
  ["section", "stream", () => goTo("stream")],
  ["section", "inspector", () => goTo("inspector")],
  ["section", "metrics", () => goTo("metrics")],
  ["section", "sim", () => goTo("simulator")],
  ["section", "pg", () => goTo("playground")],
  ["report", "r1", () => goTo("r-problem")],
  ["report", "r2", () => goTo("r-data")],
  ["report", "r3", () => goTo("r-results")],
  ["report", "r4", () => goTo("r-drivers")],
  ["report", "r5", () => goTo("r-didnt-work")],
  ["report", "r6", () => goTo("r-fairness")],
  ["report", "r7", () => goTo("r-method")],
  ["section", "credits", () => goTo("credits")],
  ["action", "play", () => togglePlay()],
  ["action", "step", () => step()],
  ["action", "restart", () => restart()],
  ["action", "thr_project", () => setIndex(nearestIndex(D, D.chosen_threshold))],
  ["action", "thr_best", () => setIndex(state.best)],
  ["action", "reset", () => resetCosts()],
  ["action", "load_fraud", () => { preset("fraud"); goTo("playground"); }],
  ["action", "theme", () => toggleTheme()],
  ["action", "lang", () => toggleLang()],
].map(([g, key, run]) => ({ group: () => t(`js.pal.group.${g}`), label: () => t(`js.pal.${key}`), run }));

const pal = $("palette");
const palInput = $("pal-input");
const palList = $("pal-list");
let palItems = [];
let palActive = 0;

function renderPalette() {
  const q = palInput.value.trim().toLowerCase();
  palItems = COMMANDS.filter((c) => !q || `${c.group()} ${c.label()}`.toLowerCase().includes(q));
  palActive = Math.min(palActive, Math.max(0, palItems.length - 1));
  palList.replaceChildren(
    ...(palItems.length
      ? palItems.map((c, i) => {
          const li = h("li");
          li.id = `pal-opt-${i}`;
          li.setAttribute("role", "option");
          li.setAttribute("aria-selected", String(i === palActive));
          li.append(h("span", null, c.label()), h("span", "grp", c.group()));
          li.addEventListener("click", () => runCommand(i));
          li.addEventListener("pointermove", () => {
            if (palActive !== i) {
              palActive = i;
              renderPalette();
            }
          });
          return li;
        })
      : [h("li", "empty", t("js.pal.empty"))]),
  );
  if (palItems.length) {
    palInput.setAttribute("aria-activedescendant", `pal-opt-${palActive}`);
    $(`pal-opt-${palActive}`).scrollIntoView({ block: "nearest" });
  } else palInput.removeAttribute("aria-activedescendant");
}

function openPalette() {
  if (pal.open) return;
  palInput.value = "";
  palActive = 0;
  renderPalette();
  pal.showModal();
  palInput.focus();
}

function runCommand(i) {
  const cmd = palItems[i];
  if (!cmd) return;
  pal.close();
  cmd.run();
}

palInput.addEventListener("input", () => {
  palActive = 0;
  renderPalette();
});
palInput.addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    const n = palItems.length || 1;
    palActive = (palActive + (e.key === "ArrowDown" ? 1 : n - 1)) % n;
    renderPalette();
  } else if (e.key === "Enter") {
    e.preventDefault();
    runCommand(palActive);
  }
});
pal.addEventListener("click", (e) => {
  if (e.target === pal) pal.close(); // backdrop
});

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    pal.open ? pal.close() : openPalette();
    return;
  }
  if (pal.open || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.target.closest("input, select, textarea, [contenteditable]")) return;
  if (e.key === "p" || e.key === "P") togglePlay();
  else if (e.key === ".") step();
  else if (e.key === "t" || e.key === "T") toggleTheme();
  else if (e.key === "l" || e.key === "L") toggleLang();
  else if (e.key === "/") {
    e.preventDefault();
    openPalette();
  }
});

// ---------------------------------------------------------------- init

function init() {
  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  $("cmd-key").textContent = isMac ? "⌘K" : "Ctrl K";
  $("open-palette").addEventListener("click", openPalette);
  $("theme-toggle").addEventListener("click", toggleTheme);
  $("lang-toggle").addEventListener("click", toggleLang);
  syncThemeButton();

  for (const node of document.querySelectorAll("button[disabled], input[disabled], select[disabled]")) node.disabled = false;
  for (const id of ["thr", "thr-mini"]) {
    const s = $(id);
    s.min = "0";
    s.max = String(D.threshold.length - 1);
    s.addEventListener("input", () => setIndex(Number(s.value)));
  }
  for (const k of PARAM_KEYS) inputs[k].addEventListener("input", onCostInput);
  $("btn-best").addEventListener("click", () => setIndex(state.best));
  $("btn-project").addEventListener("click", () => setIndex(nearestIndex(D, D.chosen_threshold)));
  $("sim-form").addEventListener("reset", (e) => {
    e.preventDefault();
    resetCosts();
  });
  $("sim-form").addEventListener("submit", (e) => e.preventDefault());

  $("s-play").addEventListener("click", togglePlay);
  $("s-step").addEventListener("click", step);
  $("s-restart").addEventListener("click", restart);
  for (const b of document.querySelectorAll(".seg [data-speed]")) b.addEventListener("click", () => setSpeed(Number(b.dataset.speed)));
  $("s-rows").addEventListener("click", (e) => {
    const b = e.target.closest("button.srow");
    if (!b) return;
    const i = Number(b.dataset.i);
    state.pinned = !(state.pinned && state.selected === i);
    state.selected = state.pinned ? i : state.played - 1;
    renderStream();
    renderInspector();
  });

  buildPlaygroundForm();
  for (const b of document.querySelectorAll("[data-preset]")) b.addEventListener("click", () => preset(b.dataset.preset));

  const vizCtx = {
    scores: SCORES,
    y: Y,
    played: () => state.played,
    thr,
    index: () => state.index,
    speed: () => state.speed,
    fmtScore,
    full: () => sampleOutcomes(SCORES, Y, AMT, thr(), state.params),
    select: selectRow,
    selectLatest: (o) => {
      for (let i = state.played - 1; i >= 0; i--) if (outcomeOf(SCORES[i], Y[i], thr()) === o) return selectRow(i);
    },
    setThreshold: (tv) => setIndex(nearestIndex(D, tv)),
    dotEvery: DATA.ui.dot_every,
    minLanePx: DATA.ui.min_lane_px,
    setIndex,
  };
  viz.flow = createFlow($("flow"), vizCtx);
  viz.strip = createStrip($("strip"), vizCtx);
  viz.conv = createConvergence($("conv"), vizCtx);
  viz.pr = createPR($("pr"), vizCtx, D);
  $("flow-static").hidden = !REDUCED;
  for (const b of document.querySelectorAll("[data-view]")) {
    b.addEventListener("click", () => {
      for (const o of document.querySelectorAll("[data-view]")) o.setAttribute("aria-checked", String(o === b));
      viz.flow.setView(b.dataset.view);
      $("flow-filter").hidden = b.dataset.view === "all";
    });
  }

  recomputeCurve();
  renderSim();
  renderPlayState();
  renderStream();
  renderCounters();
  renderInspector();
  loadPlayground(ROWS.find((r) => r.y === 1));

  let pending = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(renderSim);
  }).observe(box);
  requestAnimationFrame(loop);
}

// load the dictionary (and apply Thai if chosen) before anything is written by script
try {
  await setLang(initialLang(), { remember: false });
} catch (err) {
  console.warn("dictionary not loaded; keeping the prerendered English", err);
}
init();
