// The hero visuals must show exactly what the counters show.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { convergence, dotVisible, laneCounts, outcomeOf, prCurve, wilson } from "../src/flow.js";
import { prepareModel, score, streamRow } from "../src/model.js";
import { metricsAt, nearestIndex, sampleOutcomes } from "../src/sim.js";

const repo = fileURLToPath(new URL("../../", import.meta.url));
const page = readFileSync(repo + "web/site/index.html", "utf8");
const DATA = JSON.parse(page.match(/id="site-data">(.*?)<\/script>/s)[1].replaceAll("<\\/", "</"));
const S = DATA.stream;
const folds = Object.fromEntries(Object.entries(DATA.model.fold_models).map(([k, m]) => [k, prepareModel(m)]));
const rows = S.id.map((_, i) => streamRow(S, i));
const SCORES = rows.map((r) => score(folds[String(r.fold)], r));
const Y = rows.map((r) => r.y);
const AMT = rows.map((r) => r.amount_usd);
const P = DATA.sim.default_params;
const T0 = DATA.sim.chosen_threshold;

test("decision-flow lanes = counters, for any rows played and any threshold", () => {
  for (const t of [T0, 0.01, 0.2, 0.9]) {
    for (const n of [0, 1, 37, 500, rows.length]) {
      const lanes = laneCounts(SCORES, Y, n, t);
      const ctr = sampleOutcomes(SCORES.slice(0, n), Y.slice(0, n), AMT.slice(0, n), t, P);
      assert.deepEqual([lanes.tp, lanes.fp, lanes.fn, lanes.tn], [ctr.tp, ctr.fp, ctr.fn, ctr.tn]);
      assert.equal(lanes.stepup + lanes.approve, n);
      assert.equal(lanes.tp + lanes.fn, Y.slice(0, n).filter((v) => v === 1).length);
    }
  }
});

test("convergence chart ends at the played-so-far recall and precision", () => {
  for (const n of [50, 400, rows.length]) {
    const c = convergence(SCORES, Y, n, T0);
    const ctr = sampleOutcomes(SCORES.slice(0, n), Y.slice(0, n), AMT.slice(0, n), T0, P);
    assert.equal(c.recall.length, n);
    assert.equal(c.recall.at(-1), ctr.recall);
    assert.equal(c.precision.at(-1), ctr.precision);
    const [lo, hi] = wilson(ctr.tp, ctr.tp + ctr.fn);
    assert.equal(c.recallLo.at(-1), lo);
    assert.equal(c.recallHi.at(-1), hi);
  }
  // all rows played: the running line reaches the full-sample value
  const full = sampleOutcomes(SCORES, Y, AMT, T0, P);
  assert.equal(convergence(SCORES, Y, rows.length, T0).recall.at(-1), full.recall);
});

test("Wilson interval = the project's Python wilson_ci (fraud_e2e.stats)", () => {
  // reference values printed by fraud_e2e.stats.wilson_ci
  const ref = [
    [21, 34, 0.450410173444307, 0.7609981667549109],
    [1, 3, 0.06149194472039621, 0.7923403991979522],
    [5, 50, 0.04347576493189041, 0.21360231437479654],
    [0, 10, 0.0, 0.2775327998628892],
    [10, 10, 0.7224672001371107, 1.0],
  ];
  for (const [k, n, lo, hi] of ref) {
    const [a, b] = wilson(k, n);
    assert.ok(Math.abs(a - lo) < 1e-12 && Math.abs(b - hi) < 1e-12, `${k}/${n}: ${a} ${b}`);
  }
  assert.ok(wilson(0, 0).every(Number.isNaN));
});

test("PR-curve point at the current threshold = simulator tiles", () => {
  const pts = prCurve(DATA.sim);
  for (const t of [T0, 0.02, 0.3]) {
    const i = nearestIndex(DATA.sim, t);
    const m = metricsAt(DATA.sim, i, P);
    const p = pts.find((q) => q.i === i);
    assert.equal(p.recall, m.recall);
    assert.equal(p.precision, m.precision);
  }
});

test("'fraud + sample of legit' view keeps every fraud row and 1 in 10 legit", () => {
  const shown = rows.map((r, i) => dotVisible(i, r.y, "sample"));
  assert.ok(rows.every((r, i) => r.y !== 1 || shown[i]));
  const legit = rows.map((r, i) => [r, i]).filter(([r]) => r.y === 0);
  const frac = legit.filter(([, i]) => shown[i]).length / legit.length;
  assert.ok(Math.abs(frac - 0.1) < 0.02, String(frac));
  assert.ok(rows.every((r, i) => dotVisible(i, r.y, "all")));
  assert.equal(outcomeOf(0.5, 1, 0.5), "tp");
});
