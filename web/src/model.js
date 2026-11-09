// In-browser scoring with the exported logistic pipeline (reports/p4_web_model.json).
// Pure functions, no DOM. Mirrors fraud_e2e.export.linear_contributions / linear_scores:
//   categorical  -> coefficient of the row's category (one-hot), 0 if unseen
//   boolean      -> coef * (0|1)
//   log_scaled   -> coef * (log1p(x) - mean) / scale
//   scaled       -> coef * (x - mean) / scale
//   logit = intercept + sum(contributions);  score = 1 / (1 + exp(-logit))

export function prepareModel(m) {
  return {
    intercept: m.intercept,
    cat: m.categorical.map((b) => ({
      feature: b.feature,
      categories: b.categories,
      coef: new Map(b.categories.map((c, i) => [c, b.coef[i]])),
    })),
    bool: m.boolean,
    log: m.log_scaled,
    num: m.scaled,
  };
}

// [[feature, contribution], ...] in the pipeline's column order (cat, bool, log, num).
export function contributions(pm, row) {
  const out = [];
  for (const b of pm.cat) out.push([b.feature, b.coef.get(String(row[b.feature])) ?? 0]);
  for (const b of pm.bool) out.push([b.feature, b.coef * Number(row[b.feature])]);
  for (const b of pm.log) out.push([b.feature, (b.coef * (Math.log1p(row[b.feature]) - b.mean)) / b.scale]);
  for (const b of pm.num) out.push([b.feature, (b.coef * (row[b.feature] - b.mean)) / b.scale]);
  return out;
}

export function logit(pm, row) {
  let s = 0;
  for (const [, c] of contributions(pm, row)) s += c;
  return pm.intercept + s;
}

export function score(pm, row) {
  return 1 / (1 + Math.exp(-logit(pm, row)));
}

// Stream rows are shipped column-wise; categorical columns as codes into `levels`.
export function streamRow(stream, i) {
  const row = { id: stream.id[i], fold: stream.fold[i], y: stream.y[i] };
  for (const [f, col] of Object.entries(stream.cols)) {
    row[f] = f in stream.levels ? stream.levels[f][col[i]] : col[i];
  }
  return row;
}
