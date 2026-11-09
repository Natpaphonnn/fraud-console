// In-browser model vs Python: every transaction shipped on the page must score the same
// in JS (web/src/model.js) as in Python (fraud_e2e.export), to within 1e-9.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";

import { contributions, logit, prepareModel, score, streamRow } from "../src/model.js";

const TOL = 1e-9;
const repo = fileURLToPath(new URL("../../", import.meta.url));
const page = readFileSync(repo + "web/site/index.html", "utf8");
const block = page.match(/<script type="application\/json" id="site-data">(.*?)<\/script>/s);
const DATA = JSON.parse(block[1].replaceAll("<\\/", "</"));
const S = DATA.stream;

// Python reference: reports/p4_web_stream.csv.gz (full-precision floats; no quoted fields)
const csv = gunzipSync(readFileSync(repo + "reports/p4_web_stream.csv.gz")).toString("utf8").trim().split("\n");
const head = csv[0].split(",");
const PY = csv.slice(1).map((line) => Object.fromEntries(line.split(",").map((v, j) => [head[j], v])));

test("page ships the exported models unchanged", () => {
  const ref = JSON.parse(readFileSync(repo + "reports/p4_web_model.json", "utf8"));
  assert.deepEqual(DATA.model.fold_models, ref.fold_models);
  assert.deepEqual(DATA.model.final_model, ref.final_model);
  assert.equal(S.id.length, ref.stream.n_rows);
});

test("page rows are the Python rows, in the same order", () => {
  assert.equal(S.id.length, PY.length);
  for (let i = 0; i < PY.length; i++) {
    const r = streamRow(S, i);
    assert.equal(r.id, Number(PY[i].transaction_id), `row ${i} id`);
    assert.equal(r.y, Number(PY[i].is_fraud), `row ${i} label`);
    for (const f of DATA.model.features) {
      const py = PY[i][f];
      const js = r[f];
      if (typeof js === "string") assert.equal(js, py, `row ${i} ${f}`);
      else assert.equal(js, Number(py), `row ${i} ${f}`);
    }
  }
});

test(`JS scores match Python for every shipped row (|diff| ≤ ${TOL})`, () => {
  const folds = Object.fromEntries(Object.entries(DATA.model.fold_models).map(([k, m]) => [k, prepareModel(m)]));
  const final = prepareModel(DATA.model.final_model);
  let worstOof = 0;
  let worstFinal = 0;
  for (let i = 0; i < PY.length; i++) {
    const r = streamRow(S, i);
    worstOof = Math.max(worstOof, Math.abs(score(folds[String(r.fold)], r) - Number(PY[i].score_oof)));
    worstFinal = Math.max(worstFinal, Math.abs(score(final, r) - Number(PY[i].score_final)));
  }
  assert.ok(worstOof <= TOL, `fold-model max |diff| ${worstOof}`);
  assert.ok(worstFinal <= TOL, `final-model max |diff| ${worstFinal}`);
});

test("contributions add up to the log-odds", () => {
  const final = prepareModel(DATA.model.final_model);
  const r = streamRow(S, 0);
  const sum = contributions(final, r).reduce((s, [, c]) => s + c, final.intercept);
  assert.ok(Math.abs(sum - logit(final, r)) < 1e-12);
  assert.equal(contributions(final, r).length, DATA.model.features.length);
});

test("unseen category contributes 0 (sklearn handle_unknown='ignore')", () => {
  const final = prepareModel(DATA.model.final_model);
  const r = { ...streamRow(S, 0), channel: "Carrier pigeon" };
  const c = Object.fromEntries(contributions(final, r));
  assert.equal(c.channel, 0);
});
