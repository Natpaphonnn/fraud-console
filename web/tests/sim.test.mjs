// Node tests for the simulator logic (run: node --test web/tests/).
// The simulator must reproduce the numbers the Python pipeline committed in reports/.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  bestIndex,
  metricsAt,
  nearestIndex,
  nothingCost,
  validateParams,
} from "../src/sim.js";

const repo = fileURLToPath(new URL("../../", import.meta.url));
const readJson = (rel) => JSON.parse(readFileSync(repo + rel, "utf8"));

// The data block the page actually ships.
const page = readFileSync(repo + "web/site/index.html", "utf8");
const block = page.match(/<script type="application\/json" id="site-data">(.*?)<\/script>/s);
const D = JSON.parse(block[1].replaceAll("<\\/", "</")).sim;
const THR = readJson("reports/p2_threshold.json");
const P = D.default_params;

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);

test("page data matches reports/p2_threshold_web.json", () => {
  const w = readJson("reports/p2_threshold_web.json");
  for (const k of ["threshold", "tp", "fp", "fn", "tn", "fn_amount_usd"]) assert.deepEqual(D[k], w[k], k);
  assert.equal(D.n_rows, w.n_rows);
  assert.equal(D.n_fraud, w.n_fraud);
  assert.deepEqual(D.default_params, w.default_params);
});

test("default costs reproduce the committed operating point", () => {
  const i = nearestIndex(D, THR.threshold);
  close(D.threshold[i], THR.threshold, 1e-6, "chosen threshold is on the grid");
  assert.equal(bestIndex(D, P), i, "argmin on the web grid = project choice");
  const m = metricsAt(D, i, P);
  close(m.recall, THR.recall, 1e-4, "recall");
  close(m.precision, THR.precision, 1e-4, "precision");
  close(m.alertRate, THR.alert_rate, 1e-6, "alert rate");
  close(m.costPer1000, THR.cost_per_1000_txn, 0.05, "cost per 1000");
  close(nothingCost(D, P), THR.flag_nothing_cost_usd, 0.01, "do-nothing cost");
  close(m.saving, THR.saving_vs_flag_nothing, 1e-5, "saving");
});

test("each row of the sensitivity table is reproduced within its near-optimal cost", () => {
  const lines = readFileSync(repo + "reports/p2_cost_sensitivity.csv", "utf8").trim().split("\n");
  const head = lines[0].split(",");
  for (const line of lines.slice(1)) {
    const r = Object.fromEntries(line.split(",").map((v, j) => [head[j], v]));
    const p = { ...P };
    if (r.parameter !== "(base)") p[r.parameter] = Number(r.value);
    const m = metricsAt(D, bestIndex(D, p), p);
    // The web grid is coarser than the optimisation grid: its best cost may be slightly higher,
    // never meaningfully lower, and must stay within the 1% near-optimal band.
    const target = Number(r.cost_per_1000_txn);
    assert.ok(m.costPer1000 >= target - 0.05, `${r.parameter}×${r.factor}: ${m.costPer1000} < ${target}`);
    assert.ok(m.costPer1000 <= target * 1.01, `${r.parameter}×${r.factor}: ${m.costPer1000} ≫ ${target}`);
  }
});

test("edge thresholds behave", () => {
  const all = metricsAt(D, 0, P); // threshold 0: step up everyone
  assert.equal(D.threshold[0], 0);
  close(all.recall, 1, 1e-12, "recall at 0");
  close(all.alertRate, 1, 1e-12, "alert rate at 0");
  const none = metricsAt(D, D.threshold.length - 1, P);
  assert.ok(none.alertRate < 0.01);
});

test("parameter validation mirrors CostParams", () => {
  assert.deepEqual(validateParams(P), []);
  assert.equal(validateParams({ ...P, fp_friction_cost: -1 }).length, 1);
  assert.equal(validateParams({ ...P, fn_loss_fraction: 2.5 }).length, 1);
  assert.equal(validateParams({ ...P, review_cost_per_alert: NaN }).length, 1);
});
