// Language support. Dictionaries are built by scripts/build_web.py from web/src/i18n/*.json
// (numbers filled in from reports/). Static text carries data-i18n="key" (innerHTML) or
// data-i18n-attr="attr:key;attr2:key2"; script-generated text calls t(key, {vars}).
export const LANGS = ["en", "th"];
const STORE = "fe2e-lang";

let dict = {};
let lang = "en";

export const currentLang = () => lang;

export function t(key, vars = {}) {
  const s = dict[key];
  if (s === undefined) return key;
  return s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m));
}

// ?lang=th|en beats the saved choice, which beats the browser language; default English.
export function initialLang() {
  const q = new URLSearchParams(window.location.search).get("lang");
  if (LANGS.includes(q)) return q;
  try {
    const saved = localStorage.getItem(STORE);
    if (LANGS.includes(saved)) return saved;
  } catch {
    /* storage blocked */
  }
  return (navigator.language || "").toLowerCase().startsWith("th") ? "th" : "en";
}

export function applyStatic(root = document) {
  for (const el of root.querySelectorAll("[data-i18n]")) {
    const v = dict[el.dataset.i18n];
    if (v !== undefined) el.innerHTML = v;
  }
  for (const el of root.querySelectorAll("[data-i18n-attr]")) {
    for (const pair of el.dataset.i18nAttr.split(";")) {
      const [attr, key] = pair.split(":");
      if (dict[key] !== undefined) el.setAttribute(attr, dict[key]);
    }
  }
}

export async function setLang(next, { remember = true } = {}) {
  const res = await fetch(`i18n/${next}.json`, { cache: "force-cache" });
  if (!res.ok) throw new Error(`i18n/${next}.json: ${res.status}`);
  dict = await res.json();
  lang = next;
  document.documentElement.lang = next;
  applyStatic();
  if (remember) {
    try {
      localStorage.setItem(STORE, next);
    } catch {
      /* storage blocked */
    }
  }
  const url = new URL(window.location.href);
  if (url.searchParams.has("lang") && url.searchParams.get("lang") !== next) {
    url.searchParams.set("lang", next);
    history.replaceState(null, "", url);
  }
}
