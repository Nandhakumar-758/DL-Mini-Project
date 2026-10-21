import { useModelStatus } from "../hooks/ModelStatusContext";

type Tone = "gray" | "green" | "amber" | "red";

const TONES: Record<Tone, { dot: string; wrap: string }> = {
  gray: { dot: "bg-slate-400", wrap: "border-slate-200 bg-slate-50 text-slate-600" },
  green: { dot: "bg-emerald-500", wrap: "border-emerald-200 bg-emerald-50 text-emerald-800" },
  amber: { dot: "bg-amber-500", wrap: "border-amber-200 bg-amber-50 text-amber-900" },
  red: { dot: "bg-red-500", wrap: "border-red-200 bg-red-50 text-red-800" },
};

export function describeStatus(
  connection: ReturnType<typeof useModelStatus>["connection"],
  model: ReturnType<typeof useModelStatus>["model"],
): { label: string; tone: Tone; title: string } {
  if (connection === "checking") return { label: "Checking model…", tone: "gray", title: "Contacting the backend" };
  if (connection === "offline" || !model)
    return { label: "Backend offline", tone: "red", title: "The backend API cannot be reached." };
  if (model.ready && model.development_model)
    return { label: "Dev model ready", tone: "amber", title: model.message };
  if (model.ready) return { label: "LSTM ready", tone: "green", title: model.message };
  if (model.fallback_available)
    return { label: "No LSTM · fallback only", tone: "amber", title: model.message };
  return { label: "Model not loaded", tone: "amber", title: model.message };
}

export function ModelStatusBadge({ className = "" }: { className?: string }) {
  const { connection, model } = useModelStatus();
  const { label, tone, title } = describeStatus(connection, model);
  const t = TONES[tone];
  return (
    <div
      role="status"
      aria-live="polite"
      title={title}
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${t.wrap} ${className}`}
    >
      <span className={`h-2 w-2 rounded-full ${t.dot}`} aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}
