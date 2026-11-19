// Selectors over the platform analytics document (experiments/platform.json, served at /api/analytics and bundled at
// /data/platform.json). Pages use these helpers instead of reaching into the JSON directly.

export const TASK_LABEL = { "5class": "Focused · 5 classes", "8class": "Full · 8 classes" };

export const taskInfo = (a, task) => a?.tasks?.[task] ?? null;
export const taskClasses = (a, task) => taskInfo(a, task)?.classes ?? [];
export const modelsFor = (a, task) => (a?.models || []).filter((m) => !task || m.task === task);
export const modelByKey = (a, key) => (a?.models || []).find((m) => m.key === key) ?? null;
export const defaultModel = (a, task) => modelByKey(a, taskInfo(a, task)?.default_model);

/** Previous default (the original study's deployed model) for a task, if the document marks one. */
export const previousDefault = (a, task) => modelByKey(a, taskInfo(a, task)?.previous_default);

export function ranked(a, task, metric = "f1_macro") {
  return modelsFor(a, task)
    .filter((m) => m.test?.[metric] != null)
    .sort((x, y) => y.test[metric] - x.test[metric] || y.test.accuracy - x.test.accuracy);
}

export const comparisonsFor = (a, task) => (a?.comparisons || []).filter((c) => c.task === task);
export const recipesFor = (a, task) => (a?.cv?.recipes || []).filter((r) => r.task === task);

export const GEN_LABEL = { original: "Original study", upgrade: "Upgrade" };
export const FAMILY_COLOR = { "Deep learning": "#15834e", "Classical ML": "#d6760b" };

export function modelColor(m) {
  if (!m) return "#64748b";
  if (m.family === "Classical ML") return "#d6760b";
  return m.generation === "upgrade" ? "#15834e" : "#0891b2";
}
