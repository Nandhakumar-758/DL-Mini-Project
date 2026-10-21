import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface Props {
  title: string;
  /** one array per epoch-aligned series */
  train?: number[];
  validation?: number[];
  percent?: boolean;
  bestEpoch?: number;
  height?: number;
}

const TRAIN_COLOR = "#4f46e5"; // indigo-600
const VAL_COLOR = "#d97706"; // amber-600

export function MetricChart({ title, train, validation, percent = false, bestEpoch, height = 240 }: Props) {
  const length = Math.max(train?.length ?? 0, validation?.length ?? 0);
  const data = Array.from({ length }, (_, i) => ({
    epoch: i + 1,
    train: train?.[i],
    validation: validation?.[i],
  }));
  const fmt = (v: number) => (percent ? `${(v * 100).toFixed(1)}%` : v.toFixed(3));

  return (
    <figure className="rounded-xl border border-slate-200 bg-white p-4">
      <figcaption className="mb-2 text-sm font-semibold text-slate-800">{title}</figcaption>
      <div style={{ width: "100%", height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
            <XAxis dataKey="epoch" tick={{ fontSize: 12, fill: "#64748b" }} allowDecimals={false} />
            <YAxis
              tick={{ fontSize: 12, fill: "#64748b" }}
              tickFormatter={(v: number) => (percent ? `${(v * 100).toFixed(0)}%` : v.toFixed(1))}
              domain={["auto", "auto"]}
              width={48}
            />
            <Tooltip formatter={(v) => (typeof v === "number" ? fmt(v) : String(v))} labelFormatter={(l) => `Epoch ${l}`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {bestEpoch ? (
              <ReferenceLine x={bestEpoch} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: "best", fontSize: 11, fill: "#64748b" }} />
            ) : null}
            {train && <Line type="monotone" dataKey="train" name="Training" stroke={TRAIN_COLOR} strokeWidth={2} dot={{ r: 3 }} />}
            {validation && (
              <Line
                type="monotone"
                dataKey="validation"
                name="Validation"
                stroke={VAL_COLOR}
                strokeWidth={2}
                strokeDasharray="6 3"
                dot={{ r: 3 }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}
