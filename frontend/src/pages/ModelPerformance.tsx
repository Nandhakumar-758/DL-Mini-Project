import { Loader2 } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { Alert } from "../components/Alert";
import { MetricChart } from "../components/MetricChart";
import { describeStatus } from "../components/ModelStatusBadge";
import { useModelStatus } from "../hooks/ModelStatusContext";
import { api, ApiError } from "../services/api";
import type { Evaluation, MetricsResponse, ModelInfo } from "../types/api";

const pct = (v: number | null | undefined, digits = 2) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);
const num = (v: number | null | undefined, digits = 4) => (v == null ? "—" : v.toFixed(digits));
const int = (v: number | null | undefined) => (v == null ? "—" : v.toLocaleString("en-US"));

function Tile({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-bold tabular-nums tracking-tight text-slate-900">{value}</p>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  );
}

function Card({ title, subtitle, children }: { title: string; subtitle?: string; children: ReactNode }) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <h2 className="text-base font-semibold text-slate-900">{title}</h2>
      {subtitle && <p className="mt-0.5 text-sm text-slate-600">{subtitle}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

function KeyValueTable({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-3 border-b border-slate-100 py-1.5">
          <dt className="text-slate-500">{k}</dt>
          <dd className="text-right font-medium text-slate-900">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function LanguageTable({ evaluation }: { evaluation: Evaluation }) {
  const rows = Object.entries(evaluation.known_word);
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[480px] text-left text-sm">
        <thead className="text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="py-2 pr-3 font-medium">Language</th>
            <th className="py-2 pr-3 font-medium">Known-word samples</th>
            <th className="py-2 pr-3 font-medium">Top-1</th>
            <th className="py-2 pr-3 font-medium">Top-5</th>
            <th className="py-2 font-medium">&lt;OOV&gt; targets</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, b]) => (
            <tr key={name} className="border-t border-slate-100">
              <td className="py-2 pr-3 font-medium capitalize">{name}</td>
              <td className="py-2 pr-3 tabular-nums">{int(b.known_word_samples)}</td>
              <td className="py-2 pr-3 tabular-nums">{pct(b.known_word_top1_accuracy)}</td>
              <td className="py-2 pr-3 tabular-nums">{pct(b.known_word_top5_accuracy)}</td>
              <td className="py-2 tabular-nums">{pct(b.oov_target_share, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ModelPerformance() {
  const { connection, model } = useModelStatus();
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    Promise.all([api.metrics(ctrl.signal), api.modelInfo(ctrl.signal)])
      .then(([m, i]) => {
        setMetrics(m);
        setInfo(i);
        setError(null);
      })
      .catch((e) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setError(e instanceof ApiError ? e.message : "Could not load metrics.");
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
    return () => ctrl.abort();
  }, []);

  const status = describeStatus(connection, model);
  const training = metrics?.training ?? info?.training ?? null;
  const dataset = metrics?.dataset ?? info?.dataset ?? null;
  const bestEpoch = training?.best_epoch;
  const h = metrics?.history ?? null;
  const ev = metrics?.evaluation ?? null;
  const base = metrics?.report_baseline;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Model Performance</h1>
        <p className="mt-1 text-sm text-slate-600">
          Measured results come from the artifacts of the model on disk. The report&apos;s own results are shown
          separately, labelled as a baseline.
        </p>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-slate-600" role="status">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> Loading metrics…
        </div>
      )}

      {error && (
        <Alert kind="error" title="Could not load metrics">
          {error}
        </Alert>
      )}

      {metrics && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Tile label="Model status" value={<span className="text-lg">{status.label}</span>} hint={model?.message} />
            <Tile
              label="Vocabulary size"
              value={int(info?.tokenizer?.vocab_size)}
              hint={info?.tokenizer ? `${int(info.tokenizer.distinct_words_seen)} distinct words seen` : "Not available"}
            />
            <Tile
              label="Epochs run"
              value={training?.epochs_run ?? "—"}
              hint={bestEpoch ? `Best epoch ${bestEpoch}${training?.stopped_early ? " · early stopped" : ""}` : "Not available"}
            />
            <Tile
              label="Parameters"
              value={int(info?.architecture?.total_parameters)}
              hint={info?.architecture ? `Context length ${info.architecture.sequence_length} words` : "Not available"}
            />
          </div>

          {metrics.development_model && (
            <Alert kind="warning" title="Development model">
              These numbers come from a tiny model trained on the illustrative sample corpus. They test the pipeline and
              must not be compared with the report&apos;s results.
            </Alert>
          )}

          {!metrics.available && (
            <Alert kind="info" title="Metrics not available">
              {metrics.message} Training writes <code>history.json</code> and <code>evaluation.json</code> next to the
              model; <code>python -m scripts.evaluate_model --corpus …</code> creates <code>evaluation.json</code> for an
              existing model.
            </Alert>
          )}

          {ev && (
            <Card
              title="Measured validation results"
              subtitle={`Evaluated on ${int(ev.samples)} validation pairs${
                ev.split_strategy ? ` (${ev.split_strategy === "sentence" ? "sentence-level split" : "random n-gram split"})` : ""
              }.`}
            >
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                <Tile label="Accuracy (Top-1)" value={pct(ev.accuracy)} />
                <Tile label="Top-5 accuracy" value={pct(ev.top5_accuracy)} />
                <Tile label="Validation loss" value={num(ev.loss)} />
                <Tile label="Perplexity" value={num(ev.perplexity, 2)} hint="exp(validation loss)" />
                <Tile label="OOV targets" value={pct(ev.oov_target_share, 1)} hint="Share of targets that are <OOV>" />
              </div>
              <h3 className="mb-2 mt-6 text-sm font-semibold text-slate-800">Known-word accuracy by language</h3>
              <LanguageTable evaluation={ev} />
            </Card>
          )}

          {h && (
            <Card title="Training curves" subtitle="Per-epoch values recorded while training this model.">
              <div className="grid gap-4 lg:grid-cols-2">
                <MetricChart title="Accuracy (Top-1)" train={h.accuracy} validation={h.val_accuracy} percent bestEpoch={bestEpoch} />
                <MetricChart title="Loss" train={h.loss} validation={h.val_loss} bestEpoch={bestEpoch} />
                <MetricChart title="Top-5 accuracy" train={h.top5_accuracy} validation={h.val_top5_accuracy} percent bestEpoch={bestEpoch} />
              </div>
              {training?.restored_best_weights && (
                <p className="mt-3 text-xs text-slate-500">
                  Early stopping monitors validation loss and restores the best epoch&apos;s weights, so the saved model
                  matches the dashed &ldquo;best&rdquo; epoch, not the last one.
                </p>
              )}
            </Card>
          )}

          {(dataset || info?.config) && (
            <Card title="Dataset and artifact metadata">
              {dataset && (
                <KeyValueTable
                  rows={[
                    ["Sentences used", int(dataset.sentences as number)],
                    ["Tamil / English / mixed", `${int(dataset.tamil_sentences as number)} / ${int(dataset.english_sentences as number)} / ${int(dataset.mixed_sentences as number)}`],
                    ["Total words", int(dataset.total_words as number)],
                    ["Unique words", int(dataset.unique_words as number)],
                    ["Duplicates removed", int(dataset.duplicates_removed as number)],
                    ["Training / validation pairs", `${int(dataset.train_pairs as number)} / ${int(dataset.val_pairs as number)}`],
                    ["Split strategy", String(dataset.split_strategy ?? "—")],
                    ["Sources", Array.isArray(dataset.sources) ? dataset.sources.join(", ") : "—"],
                  ]}
                />
              )}
              {info?.config && (
                <div className={dataset ? "mt-5" : ""}>
                  <h3 className="mb-2 text-sm font-semibold text-slate-800">Training configuration</h3>
                  <KeyValueTable
                    rows={[
                      ["Embedding dimension", String(info.config.embedding_dim ?? "—")],
                      ["LSTM units (×2 layers)", String(info.config.lstm_units ?? "—")],
                      ["Dense units", String(info.config.dense_units ?? "—")],
                      ["Dropout", String(info.config.dropout ?? "—")],
                      ["Batch size", String(info.config.batch_size ?? "—")],
                      ["Max epochs / patience", `${info.config.max_epochs ?? "—"} / ${info.config.patience ?? "—"}`],
                      ["Learning rate", String(info.config.learning_rate ?? "—")],
                      ["Run", `${info.run?.run_id ?? "—"} · ${info.run?.trained_at ?? "—"}`],
                    ]}
                  />
                </div>
              )}
            </Card>
          )}

          {base && (
            <Card title={base.label} subtitle="Historical results quoted from the report. They were not measured by the model that is currently loaded.">
              <Alert kind="info" className="mb-4">
                {base.note}
              </Alert>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                <Tile label="Accuracy (Top-1)" value={pct(base.evaluation.accuracy)} />
                <Tile label="Top-5 accuracy" value={pct(base.evaluation.top5_accuracy)} />
                <Tile label="Validation loss" value={num(base.evaluation.loss)} />
                <Tile label="Perplexity" value={num(base.evaluation.perplexity, 2)} />
                <Tile label="OOV targets" value={pct(base.evaluation.oov_target_share, 2)} />
              </div>
              <div className="mt-5 grid gap-4 lg:grid-cols-2">
                <MetricChart title="Report: loss per epoch" train={base.history.loss} validation={base.history.val_loss} bestEpoch={base.training.best_epoch} height={200} />
                <MetricChart title="Report: top-5 accuracy per epoch" train={base.history.top5_accuracy} validation={base.history.val_top5_accuracy} percent bestEpoch={base.training.best_epoch} height={200} />
              </div>
              <h3 className="mb-2 mt-6 text-sm font-semibold text-slate-800">Report: known-word accuracy by language</h3>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[480px] text-left text-sm">
                  <thead className="text-xs uppercase tracking-wide text-slate-500">
                    <tr>
                      <th className="py-2 pr-3 font-medium">Language</th>
                      <th className="py-2 pr-3 font-medium">Known-word samples</th>
                      <th className="py-2 pr-3 font-medium">Top-1</th>
                      <th className="py-2 pr-3 font-medium">Top-5</th>
                      <th className="py-2 font-medium">&lt;OOV&gt; targets</th>
                    </tr>
                  </thead>
                  <tbody>
                    {base.known_word.map((r) => (
                      <tr key={r.language} className="border-t border-slate-100">
                        <td className="py-2 pr-3 font-medium">{r.language}</td>
                        <td className="py-2 pr-3 tabular-nums">{int(r.samples)}</td>
                        <td className="py-2 pr-3 tabular-nums">{pct(r.top1)}</td>
                        <td className="py-2 pr-3 tabular-nums">{pct(r.top5)}</td>
                        <td className="py-2 tabular-nums">{pct(r.oov_targets, 1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-3 text-xs text-slate-500">
                Report corpus: {int(base.corpus.total_sentences)} sentences ({int(base.corpus.tamil_sentences)} Tamil,{" "}
                {int(base.corpus.english_sentences)} English), {int(base.corpus.training_pairs)} training pairs; trained {base.training.epochs_run} epochs,
                best epoch {base.training.best_epoch}.
              </p>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
