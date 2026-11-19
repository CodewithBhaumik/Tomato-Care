import { useMemo } from "react";
import { Cpu, Layers3, Sparkles } from "lucide-react";
import { useData } from "../../context/DataContext.jsx";
import { ms, pct } from "../../lib/format.js";
import { Badge, Segmented } from "../ui/primitives.jsx";

export const TASK_OPTIONS = [
  { value: "5class", label: "Focused · 5 classes" },
  { value: "8class", label: "Full · 8 classes" },
];

/** Task scope + model selection. Upgraded models are listed first; the task default is pre-selected. */
export default function ModelPicker({ task, onTask, model, onModel, disabled }) {
  const { models } = useData();
  const list = useMemo(() => (models?.models || []).filter((m) => m.task === task), [models, task]);
  const selected = list.find((m) => m.key === model) || null;

  return (
    <div className="flex flex-col gap-4">
      <div>
        <p className="mb-2 text-sm font-medium text-ink">Scope</p>
        <Segmented options={TASK_OPTIONS} value={task} onChange={onTask} label="Model scope" />
        <p className="mt-2 text-xs leading-relaxed text-muted">
          {task === "5class"
            ? "Healthy, Leaf Miner, Magnesium and Nitrogen Deficiency, Spotted Wilt Virus — the classes with enough photos to be recognised reliably."
            : "All 8 classes, including Early Blight, Late Blight and Potassium Deficiency (few photos: lower reliability)."}
        </p>
      </div>
      <div>
        <label htmlFor="model-select" className="mb-2 block text-sm font-medium text-ink">
          Model
        </label>
        <select
          id="model-select"
          value={model || ""}
          onChange={(e) => onModel(e.target.value)}
          disabled={disabled || !list.length}
          className="h-10 w-full rounded-xl border border-line-strong bg-surface px-3 text-sm text-ink shadow-sm outline-none focus:border-leaf-500 focus:ring-2 focus:ring-leaf-500/20"
        >
          {!list.length && <option value="">Models load when the server is online</option>}
          {list.map((m) => (
            <option key={m.key} value={m.key}>
              {m.name}
              {models?.defaults_by_task?.[task] === m.key ? " — recommended" : ""}
            </option>
          ))}
        </select>
        {selected && (
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {selected.generation === "upgrade" ? <Badge tone="leaf" icon={Sparkles}>Upgraded model</Badge> : <Badge tone="slate">Original study</Badge>}
            <Badge tone="slate" icon={Layers3}>{selected.family}</Badge>
            {selected.test_metrics?.accuracy != null && <Badge tone="sky">test acc {pct(selected.test_metrics.accuracy)}</Badge>}
            {selected.latency_ms != null && <Badge tone="slate" icon={Cpu}>{ms(selected.latency_ms)} CPU</Badge>}
          </div>
        )}
      </div>
    </div>
  );
}
