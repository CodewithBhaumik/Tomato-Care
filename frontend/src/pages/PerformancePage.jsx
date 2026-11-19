import { useEffect, useMemo, useState } from "react";
import { Activity, BarChart3, Grid3x3, LineChart as LineIcon, ScanEye, ShieldCheck, Target } from "lucide-react";
import { ReliabilityDiagram, RiskCoverage, TrainingCurves } from "../components/charts/charts.jsx";
import ConfusionMatrix from "../components/charts/ConfusionMatrix.jsx";
import { useDocumentTitle } from "../components/layout/chrome.jsx";
import { Callout, Card, CardHeader, ErrorState, PageHeader, Segmented, Skeleton, Stat } from "../components/ui/primitives.jsx";
import { useData } from "../context/DataContext.jsx";
import { defaultModel, modelByKey, modelsFor, recipesFor, taskClasses, taskInfo } from "../lib/analytics.js";
import { classMeta } from "../lib/classes.js";
import { ci, cx, num, pct } from "../lib/format.js";

function PerClassTable({ model }) {
  return (
    <div className="overflow-x-auto scrollbar-thin">
      <table className="w-full min-w-[520px] text-sm">
        <thead className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
          <tr>
            <th className="px-5 py-3 font-medium">Class</th>
            <th className="px-3 py-3 text-right font-medium">Precision</th>
            <th className="px-3 py-3 text-right font-medium">Recall</th>
            <th className="px-3 py-3 text-right font-medium">F1</th>
            <th className="px-5 py-3 text-right font-medium">Test photos</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {model.per_class.map((c) => {
            const meta = classMeta(c.key);
            return (
              <tr key={c.key}>
                <td className="px-5 py-2.5">
                  <span className="flex items-center gap-2">
                    <span className="size-2.5 rounded-full" style={{ background: meta.color }} />
                    <span className="font-medium text-ink">{meta.name}</span>
                  </span>
                </td>
                {["precision", "recall", "f1"].map((k) => (
                  <td key={k} className="px-3 py-2.5 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <div className="hidden h-1.5 w-16 overflow-hidden rounded-full bg-surface-3 sm:block">
                        <div className="h-full rounded-full" style={{ width: `${(c[k] || 0) * 100}%`, background: meta.color }} />
                      </div>
                      <span className="w-12 font-mono tabular text-ink-2">{pct(c[k], 0)}</span>
                    </div>
                  </td>
                ))}
                <td className={cx("px-5 py-2.5 text-right font-mono tabular", c.support < 10 ? "text-sun-600" : "text-muted")}>{c.support}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function PerformancePage() {
  useDocumentTitle("Performance");
  const { analytics, analyticsLoading, analyticsError, reloadAnalytics } = useData();
  const [task, setTask] = useState("5class");
  const [key, setKey] = useState(null);

  useEffect(() => {
    if (analytics) setKey(defaultModel(analytics, task)?.key || modelsFor(analytics, task)[0]?.key);
  }, [analytics, task]);

  const model = modelByKey(analytics, key);
  const classes = taskClasses(analytics, task);
  const selective = analytics?.selective?.[task];
  const recipes = recipesFor(analytics, task);
  const curvesRecipe = useMemo(() => {
    const keyRecipe = model?.recipe;
    return recipes.find((r) => r.recipe === keyRecipe) || recipes.find((r) => r.curves?.length) || null;
  }, [recipes, model]);
  const lowSupport = model?.per_class?.filter((c) => c.support < 10) || [];

  if (analyticsError && !analytics) return <ErrorState title="Could not load results" error={analyticsError} onRetry={reloadAnalytics} />;

  return (
    <div>
      <PageHeader
        eyebrow="Interactive results"
        title="Performance"
        description="A deep dive into one model at a time: confusion matrix, per-class precision and recall, calibration of its confidence, the accuracy–coverage trade-off and how it trained."
        actions={<Segmented options={[{ value: "5class", label: "5 classes" }, { value: "8class", label: "8 classes" }]} value={task} onChange={setTask} label="Task" />}
      />
      {analyticsLoading && !analytics && <Skeleton className="h-96 rounded-2xl" />}
      {analytics && (
        <div className="flex flex-col gap-6">
          <Card className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center">
            <label htmlFor="perf-model" className="text-sm font-medium text-ink">
              Model
            </label>
            <select
              id="perf-model"
              value={key || ""}
              onChange={(e) => setKey(e.target.value)}
              className="h-10 flex-1 rounded-xl border border-line-strong bg-surface px-3 text-sm text-ink outline-none focus:border-leaf-500 focus:ring-2 focus:ring-leaf-500/20"
            >
              {modelsFor(analytics, task).map((m) => (
                <option key={m.key} value={m.key}>
                  {m.name} {m.key === taskInfo(analytics, task)?.default_model ? "(default)" : ""} — {m.generation === "upgrade" ? "upgrade" : "original"}
                </option>
              ))}
            </select>
          </Card>

          {model && (
            <>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <Stat icon={Target} label="Test accuracy" value={pct(model.test.accuracy)} hint={`95% CI ${ci(model.test.accuracy_ci)}`} />
                <Stat icon={BarChart3} tone="violet" label="Macro-F1" value={pct(model.test.f1_macro)} hint={`95% CI ${ci(model.test.f1_macro_ci)}`} />
                <Stat icon={Activity} tone="sky" label="Calibration error (ECE)" value={num(model.test.ece, 3)} hint={model.calibration ? `temperature T = ${num(model.calibration.temperature, 2)}` : "probabilities from the classifier"} />
                <Stat
                  icon={ShieldCheck}
                  tone="sun"
                  label="Selective accuracy"
                  value={model.selective ? pct(model.selective.test?.accuracy_kept ?? model.selective.accuracy_kept) : "—"}
                  hint={model.selective ? `answers ${pct(model.selective.test?.coverage ?? model.selective.coverage, 0)} of photos at threshold ${pct(model.selective.threshold, 0)}` : "no selective threshold"}
                />
              </div>

              {lowSupport.length > 0 && (
                <Callout tone="sun" title="Small classes">
                  {lowSupport.map((c) => `${classMeta(c.key).name} (${c.support})`).join(", ")} have fewer than 10 test photos: their per-class scores can swing by 10–50 points with a single photo. Read them together with the confidence intervals.
                </Callout>
              )}

              <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
                <Card className="p-5">
                  <div className="mb-1 flex items-center gap-2 text-sm font-semibold text-ink">
                    <Grid3x3 className="size-4 text-leaf-600" aria-hidden /> Confusion matrix (test)
                  </div>
                  <ConfusionMatrix matrix={model.confusion} classes={classes} />
                  <div className="mt-3 flex flex-wrap justify-center gap-x-3 gap-y-1 text-[11px] text-muted">
                    {classes.map((k) => (
                      <span key={k}>
                        <strong style={{ color: classMeta(k).color }}>{classMeta(k).abbr}</strong> {classMeta(k).short}
                      </span>
                    ))}
                  </div>
                </Card>
                <Card>
                  <CardHeader icon={BarChart3} title="Per-class results" subtitle="Precision: when it says this class, how often it is right. Recall: how many photos of this class it finds." />
                  <PerClassTable model={model} />
                </Card>
              </div>

              <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                <Card>
                  <CardHeader icon={Activity} title="Calibration (reliability diagram)" subtitle="Bars on the dashed diagonal = the stated confidence matches the actual accuracy" />
                  <div className="p-4">
                    {model.calibration?.reliability_after ? (
                      <ReliabilityDiagram before={model.calibration.reliability_before} after={model.calibration.reliability_after} />
                    ) : (
                      <p className="p-6 text-center text-sm text-muted">No calibration data for this model.</p>
                    )}
                    {model.calibration && (
                      <p className="mt-2 text-center text-xs text-muted">
                        ECE {num(model.calibration.ece_before, 3)} before → {num(model.calibration.ece_after, 3)} after temperature scaling (fitted on {model.calibration.fitted_on || "held-out training data"}).
                      </p>
                    )}
                  </div>
                </Card>
                <Card>
                  <CardHeader icon={ScanEye} title="Accuracy vs coverage" subtitle="Raising the confidence threshold answers fewer photos but gets more of them right" />
                  <div className="p-4">
                    {selective && selective.model === model.key ? (
                      <>
                        <RiskCoverage curve={selective.curve} chosen={selective.chosen} />
                        <p className="mt-2 text-center text-xs text-muted">
                          Chosen threshold {pct(selective.chosen.threshold, 0)} ({selective.rule}). On test: answers {pct(selective.chosen.coverage, 0)}, accuracy {pct(selective.chosen.accuracy_kept)}.
                        </p>
                      </>
                    ) : (
                      <p className="p-6 text-center text-sm text-muted">The risk–coverage curve is computed for the default model of each task.</p>
                    )}
                  </div>
                </Card>
              </div>

              {curvesRecipe?.curves?.length > 0 && (
                <Card>
                  <CardHeader icon={LineIcon} title={`Training curves · ${curvesRecipe.label}`} subtitle="Mean over the 5 cross-validation folds; held-out scores are monitored only, never used to stop training or pick an epoch" />
                  <div className="p-4">
                    <TrainingCurves curves={curvesRecipe.curves} />
                  </div>
                </Card>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
