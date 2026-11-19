import { useMemo } from "react";
import { Cpu, Layers3, ShieldCheck, Sparkles } from "lucide-react";
import { useData } from "../../context/DataContext.jsx";
import { ms, pct } from "../../lib/format.js";
import { Badge, Segmented } from "../ui/primitives.jsx";

export const TASK_OPTIONS = [
  { value: "5class", label: "Focused · 5 classes" },
  { value: "8class", label: "Full · 8 classes" },
];

function ModelBadges({ m }) {
  return (
    <div className="mt-2.5 flex flex-wrap gap-1.5">
      {m.generation === "upgrade" ? <Badge tone="leaf" icon={Sparkles}>Upgraded model</Badge> : <Badge tone="slate">Original study</Badge>}
      <Badge tone="slate" icon={Layers3}>{m.family}</Badge>
      {m.test_metrics?.accuracy != null && <Badge tone="sky">test acc {pct(m.test_metrics.accuracy)}</Badge>}
      {m.confidence_threshold != null && <Badge tone="slate" icon={ShieldCheck}>answers at ≥ {pct(m.confidence_threshold, 0)}</Badge>}
      {m.latency_ms != null && <Badge tone="slate" icon={Cpu}>{ms(m.latency_ms)} CPU</Badge>}
    </div>
  );
}

/** Task scope + the model that answers for it. One validated model per scope: shown as a card, not a choice. */
export default function ModelPicker({ task, onTask, model, onModel, disabled }) {
  const { models } = useData();
  const list = useMemo(() => (models?.models || []).filter((m) => m.task === task), [models, task]);
  const selected = list.find((m) => m.key === model) || list[0] || null;

  return (
    <div className="flex flex-col gap-4">
      <div>
        <p className="mb-2 text-sm font-medium text-ink">Scope</p>
        <Segmented options={TASK_OPTIONS} value={task} onChange={onTask} label="Model scope" />
        <p className="mt-2 text-xs leading-relaxed text-muted">
          {task === "5class"
            ? "Healthy, Leaf Miner, Magnesium and Nitrogen Deficiency, Spotted Wilt Virus — the classes with enough photos to be recognised reliably. Other conditions are answered “Uncertain”."
            : "All 8 classes, including Early Blight, Late Blight and Potassium Deficiency (few photos: lower reliability)."}
        </p>
      </div>
      <div>
        <p className="mb-2 text-sm font-medium text-ink">Model</p>
        {list.length > 1 ? (
          <select
            id="model-select"
            aria-label="Model"
            value={selected?.key || ""}
            onChange={(e) => onModel(e.target.value)}
            disabled={disabled}
            className="h-10 w-full rounded-xl border border-line-strong bg-surface px-3 text-sm text-ink shadow-sm outline-none focus:border-leaf-500 focus:ring-2 focus:ring-leaf-500/20"
          >
            {list.map((m) => (
              <option key={m.key} value={m.key}>
                {m.name}
                {models?.defaults_by_task?.[task] === m.key ? " — recommended" : ""}
              </option>
            ))}
          </select>
        ) : selected ? (
          <div className="rounded-xl border border-line-strong bg-surface px-3.5 py-2.5 text-sm" data-testid="model-card">
            <p className="font-medium text-ink">{selected.name}</p>
            {selected.description && <p className="mt-0.5 text-xs leading-relaxed text-muted">{selected.description}</p>}
          </div>
        ) : (
          <p className="rounded-xl border border-dashed border-line-strong px-3.5 py-2.5 text-sm text-muted">Models load when the server is online</p>
        )}
        {selected && <ModelBadges m={selected} />}
      </div>
    </div>
  );
}
