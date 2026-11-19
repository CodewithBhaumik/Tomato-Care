import { CheckCircle2, ShieldAlert, ShieldCheck, Sparkles } from "lucide-react";
import { Badge, Callout, Card, CardHeader } from "../ui/primitives.jsx";
import { ci, cx, num, pct, signedPct } from "../../lib/format.js";

const STRESS_LABEL = {
  blur_sigma2: "Blur (σ 2)",
  blur_sigma4: "Strong blur (σ 4)",
  "dark_x0.35": "Very dark",
  "overexposed_x1.7": "Over-exposed",
  jpeg_q15: "Heavy JPEG (q15)",
  background_black: "Black background",
  background_wood: "Wood background",
  background_grass: "Grass background",
  background_fabric: "Fabric background",
  four_leaves_same: "4 leaves, same condition",
  four_leaves_mixed: "4 leaves, mixed",
};
const OOD_LABEL = {
  other_crop: "Other crops (PlantVillage)",
  tomato_unsupported: "Tomato diseases not in the dataset",
  everyday: "Everyday photos",
  probes: "Synthetic non-leaf images",
  empty_paper: "Empty paper (leaf removed)",
};

function Th({ children, right = false, className = "" }) {
  return <th className={cx("px-3 py-2.5 font-medium", right && "text-right", className)}>{children}</th>;
}
function Td({ children, right = true, className = "" }) {
  return <td className={cx("px-3 py-2.5", right && "text-right font-mono tabular", className)}>{children}</td>;
}

/** Share bar: answered correctly (green), answered wrongly (red), refused (grey). */
function OutcomeBar({ o }) {
  if (!o) return <span className="text-muted">—</span>;
  const right = o.answered_correct ?? 0;
  const wrong = o.answered_wrong ?? (o.answered_correct == null ? o.answered : 0);
  const refused = o.refused ?? 0;
  return (
    <div className="flex min-w-[140px] items-center gap-2">
      <div className="flex h-2 flex-1 overflow-hidden rounded-full bg-surface-2" aria-hidden>
        <span className="bg-leaf-500" style={{ width: `${100 * right}%` }} />
        <span className="bg-tomato-500" style={{ width: `${100 * wrong}%` }} />
        <span className="bg-line-strong" style={{ width: `${100 * refused}%` }} />
      </div>
      <span className="w-24 text-right font-mono text-xs tabular text-muted">
        {o.answered_correct != null ? `${pct(right, 0)} · ` : ""}
        <span className="text-tomato-600">{pct(wrong, 0)}</span>
      </span>
    </div>
  );
}

/** The final round: the imbalance arms compared out-of-fold, the chosen pipeline, and before vs after on test. */
export default function FinalPipeline({ final, task }) {
  const t = final?.tasks?.[task];
  if (!t) return null;
  const finalRow = t.test?.find((r) => r.final);
  const before = t.test?.find((r) => !r.final);
  const pair = finalRow && before ? [before.key, finalRow.key] : [];
  const stressNames = Object.keys(t.stress?.[finalRow?.key] || {});
  const oodNames = Object.keys(OOD_LABEL).filter((k) => t.out_of_dataset?.[finalRow?.key]?.[k]);
  return (
    <Card>
      <CardHeader
        icon={Sparkles}
        title="Final pipeline — one validated model per mode"
        subtitle="Imbalance handling, calibration, threshold and unusual-image limits chosen on out-of-fold data only; the test photos were scored once at the end"
      />
      <div className="flex flex-col gap-6 p-5">
        <div>
          <p className="mb-2 text-sm font-semibold text-ink">1 · Imbalance arms (out-of-fold, cross-fitted)</p>
          <div className="overflow-x-auto scrollbar-thin rounded-xl border border-line">
            <table className="w-full min-w-[860px] text-sm">
              <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
                <tr>
                  <Th>Arm · training</Th>
                  <Th right>Macro-F1</Th>
                  <Th right>Accuracy</Th>
                  <Th right>NLL</Th>
                  <Th right>ECE</Th>
                  <Th>Calibration</Th>
                  <Th right>Threshold</Th>
                  <Th right>Answered</Th>
                  <Th right>Acc. answered</Th>
                  {task === "5class" && <Th right>Unsupported refused</Th>}
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {t.arms.map((a) => (
                  <tr key={a.arm} className={a.chosen ? "bg-leaf-500/5" : "bg-surface"}>
                    <td className="px-3 py-2.5">
                      <span className="font-medium text-ink">{a.arm}</span> <span className="text-muted">· {a.label}</span>
                      {a.tau ? <span className="ml-1 text-xs text-muted">(τ {num(a.tau, 2)})</span> : null}
                      {a.chosen && <Badge tone="leaf" icon={CheckCircle2} className="ml-2">chosen</Badge>}
                    </td>
                    <Td>{pct(a.oof.f1_macro)}</Td>
                    <Td>{pct(a.oof.accuracy)}</Td>
                    <Td>{num(a.oof.nll, 3)}</Td>
                    <Td>{num(a.oof.ece, 3)}</Td>
                    <td className="px-3 py-2.5 text-xs text-muted">
                      {a.calibration} <span className="font-mono">(NLL {num(a.nll_temperature, 3)} vs {num(a.nll_vector, 3)})</span>
                    </td>
                    <Td>{num(a.threshold, 2)}</Td>
                    <Td>{pct(a.coverage)}</Td>
                    <Td>{pct(a.accuracy_kept)}</Td>
                    {task === "5class" && <Td>{pct(a.out_of_scope_refusal)}</Td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted">
            Chosen: arm {t.chosen} — {t.why}. Unusual-image gate: {t.gate?.method} feature distance (out-of-fold AUROC {num(t.gate?.auroc_select, 3)}).
          </p>
        </div>

        {t.test && (
          <div>
            <p className="mb-2 text-sm font-semibold text-ink">2 · Before vs after on the held-out test photos</p>
            <div className="overflow-x-auto scrollbar-thin rounded-xl border border-line">
              <table className="w-full min-w-[900px] text-sm">
                <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
                  <tr>
                    <Th>Model</Th>
                    <Th right>Accuracy (95% CI)</Th>
                    <Th right>Macro-F1</Th>
                    <Th right>Weighted F1</Th>
                    <Th right>ECE</Th>
                    <Th right>Brier</Th>
                    <Th right>Threshold</Th>
                    <Th right>Answered → acc.</Th>
                    {task === "5class" && <Th right>Unsupported: wrong answer</Th>}
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {t.test.map((r) => (
                    <tr key={r.key} className={r.final ? "bg-leaf-500/5" : "bg-surface"}>
                      <td className="px-3 py-2.5">
                        <span className={cx("font-medium", r.final ? "text-ink" : "text-muted")}>{r.name}</span>
                        {r.final && <Badge tone="leaf" className="ml-2">served</Badge>}
                      </td>
                      <Td>
                        {pct(r.accuracy)} <span className="text-xs text-muted">({ci(r.accuracy_ci, 0)})</span>
                      </Td>
                      <Td>{pct(r.f1_macro)}</Td>
                      <Td>{pct(r.f1_weighted)}</Td>
                      <Td>{num(r.ece, 3)}</Td>
                      <Td>{num(r.brier, 3)}</Td>
                      <Td>{num(r.threshold, 2)}</Td>
                      <Td>
                        {pct(r.coverage, 0)} → {pct(r.accuracy_kept)}
                      </Td>
                      {task === "5class" && <Td>{pct(r.out_of_scope?.answered_wrong, 0)}</Td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {t.comparisons?.length > 0 && (
              <p className="mt-2 text-xs text-muted">
                Paired bootstrap, final vs {t.comparisons[0].b_name}: Δ accuracy {signedPct(t.comparisons[0].accuracy_diff)} (95% CI{" "}
                {signedPct(t.comparisons[0].accuracy_ci[0])} … {signedPct(t.comparisons[0].accuracy_ci[1])}), Δ macro-F1 {signedPct(t.comparisons[0].f1_diff)} (
                {signedPct(t.comparisons[0].f1_ci[0])} … {signedPct(t.comparisons[0].f1_ci[1])}).
              </p>
            )}
          </div>
        )}

        {pair.length === 2 && stressNames.length > 0 && (
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            <div>
              <p className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-ink">
                <ShieldCheck className="size-4 text-muted" aria-hidden /> 3 · Stress tests (test photos)
              </p>
              <div className="overflow-x-auto scrollbar-thin rounded-xl border border-line">
                <table className="w-full min-w-[460px] text-sm">
                  <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
                    <tr>
                      <Th>Condition</Th>
                      <Th>Before</Th>
                      <Th>Final</Th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line">
                    {stressNames.map((n) => (
                      <tr key={n} className="bg-surface">
                        <td className="px-3 py-2 text-ink">{STRESS_LABEL[n] || n}</td>
                        {pair.map((k) => (
                          <td key={k} className="px-3 py-2">
                            <OutcomeBar o={t.stress?.[k]?.[n]} />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div>
              <p className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-ink">
                <ShieldAlert className="size-4 text-muted" aria-hidden /> 4 · Out-of-dataset images: share refused
              </p>
              <div className="overflow-x-auto scrollbar-thin rounded-xl border border-line">
                <table className="w-full min-w-[420px] text-sm">
                  <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
                    <tr>
                      <Th>Images</Th>
                      <Th right>Before</Th>
                      <Th right>Final</Th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line">
                    {oodNames.map((n) => (
                      <tr key={n} className="bg-surface">
                        <td className="px-3 py-2 text-ink">
                          {OOD_LABEL[n]} <span className="text-xs text-muted">({t.out_of_dataset[finalRow.key][n].photos})</span>
                        </td>
                        {pair.map((k) => (
                          <Td key={k}>{pct(t.out_of_dataset?.[k]?.[n]?.refused, 0)}</Td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-2 text-xs text-muted">Bars: answered correctly · answered wrongly (red %) · refused. Refused = “Uncertain” or “Unusual image”.</p>
            </div>
          </div>
        )}
        {final.note && (
          <Callout tone="slate" title="How the test set was used">
            {final.note}
          </Callout>
        )}
      </div>
    </Card>
  );
}
