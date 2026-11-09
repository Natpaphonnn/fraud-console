// Applies the saved theme before first paint (dark is the default). Loaded as a plain script.
try {
  const t = localStorage.getItem("fe2e-theme");
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
} catch {
  /* storage blocked: keep the default */
}
