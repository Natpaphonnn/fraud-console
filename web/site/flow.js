// Pure computations behind the hero visuals (decision flow, convergence chart, PR curve).
// No DOM. Every visual reads the same arrays as the counters (scores, labels, rows played,
// threshold), so its final values equal the counters; web/tests/flow.test.mjs checks that.

// Outcome of one transaction at threshold t.
export function outcomeOf(score, y, t) {
  const step = score >= t;
  return y === 1 ? (step ? "tp" : "fn") : step ? "fp" : "tn";
}

// Counts over the first n rows (the rows played so far).
export function laneCounts(scores, y, n, t) {
  const c = { tp: 0, fp: 0, fn: 0, tn: 0 };
  for (let i = 0; i < n; i++) c[outcomeOf(scores[i], y[i], t)]++;
  return { ...c, n, stepup: c.tp + c.fp, approve: c.fn + c.tn };
}

// Wilson score interval for k successes out of n (95% by default); [NaN, NaN] when n = 0.
export function wilson(k, n, z = 1.959963984540054) {
  if (!n) return [NaN, NaN];
  const p = k / n;
  const z2 = z * z;
  const den = 1 + z2 / n;
  const centre = (p + z2 / (2 * n)) / den;
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / den;
  return [k === 0 ? 0 : Math.max(0, centre - half), k === n ? 1 : Math.min(1, centre + half)];
}

// Running recall and precision after each of the first n rows, with Wilson bands.
// Entry j describes rows 0..j. NaN where undefined (no fraud yet / nothing stepped up yet).
export function convergence(scores, y, n, t) {
  const out = { recall: [], recallLo: [], recallHi: [], precision: [], precLo: [], precHi: [] };
  let tp = 0;
  let fp = 0;
  let fn = 0;
  for (let i = 0; i < n; i++) {
    const o = outcomeOf(scores[i], y[i], t);
    if (o === "tp") tp++;
    else if (o === "fp") fp++;
    else if (o === "fn") fn++;
    const [rl, rh] = wilson(tp, tp + fn);
    const [pl, ph] = wilson(tp, tp + fp);
    out.recall.push(tp + fn ? tp / (tp + fn) : NaN);
    out.recallLo.push(rl);
    out.recallHi.push(rh);
    out.precision.push(tp + fp ? tp / (tp + fp) : NaN);
    out.precLo.push(pl);
    out.precHi.push(ph);
  }
  return out;
}

// Precision-recall points along the simulator's threshold grid (same counts as the cost curve).
export function prCurve(d) {
  const pts = [];
  for (let i = 0; i < d.threshold.length; i++) {
    const flagged = d.tp[i] + d.fp[i];
    if (flagged > 0) pts.push({ i, recall: d.tp[i] / (d.tp[i] + d.fn[i]), precision: d.tp[i] / flagged });
  }
  return pts;
}

// Which rows get an animated dot in the "fraud + sample of legit" view: every fraud row and
// one legit row in `every` (by position in the stream, deterministic). Counts are unaffected.
export function dotVisible(i, y, view, every = 10) {
  return view === "all" || y === 1 || i % every === 0;
}
