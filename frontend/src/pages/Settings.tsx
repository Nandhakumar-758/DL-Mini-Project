import { History, Loader2, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Alert } from "../components/Alert";
import { describeStatus } from "../components/ModelStatusBadge";
import { PrivacyNotice } from "../components/PrivacyNotice";
import { useModelStatus } from "../hooks/ModelStatusContext";
import { api, ApiError } from "../services/api";
import type { HistoryResponse } from "../types/api";

function errorMessage(e: unknown): string {
  return e instanceof ApiError ? e.message : "Something went wrong.";
}

export function Settings() {
  const { connection, model, refresh } = useModelStatus();
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setData(await api.history());
      setError(null);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const toggle = async (enabled: boolean) => {
    setBusy(true);
    setNotice(null);
    try {
      await api.setHistoryEnabled(enabled);
      await load();
      setNotice(enabled ? "History is on. New predictions will be saved on this server." : "History is off. Nothing new will be saved.");
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const clear = async () => {
    setBusy(true);
    try {
      const { deleted } = await api.clearHistory();
      setConfirming(false);
      setNotice(`Deleted ${deleted} saved ${deleted === 1 ? "entry" : "entries"}.`);
      await load();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const status = describeStatus(connection, model);
  const enabled = data?.enabled ?? false;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Settings</h1>
        <p className="mt-1 text-sm text-slate-600">Control what is saved and see how the backend is configured.</p>
      </div>

      {error && (
        <Alert kind="error" title="Request failed">
          {error}
        </Alert>
      )}
      {notice && !error && <Alert kind="success">{notice}</Alert>}

      <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5" aria-labelledby="history-heading">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 id="history-heading" className="flex items-center gap-2 text-base font-semibold text-slate-900">
              <History className="h-4 w-4 text-indigo-600" aria-hidden="true" /> Prediction history
            </h2>
            <p className="mt-1 max-w-xl text-sm text-slate-600">
              Off by default. When on, the text you submit and the suggested words are stored in a local SQLite database
              on the backend so you can review them here. You can clear everything at any time.
            </p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={enabled}
            aria-label="Save prediction history"
            disabled={busy || loading || !data}
            onClick={() => void toggle(!enabled)}
            className={`relative mt-1 h-6 w-11 shrink-0 rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:opacity-50 ${
              enabled ? "bg-indigo-600" : "bg-slate-300"
            }`}
          >
            <span
              className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${
                enabled ? "translate-x-5" : ""
              }`}
            />
          </button>
        </div>

        <div className="mt-4 border-t border-slate-100 pt-4">
          {loading && (
            <p className="flex items-center gap-2 text-sm text-slate-600">
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> Loading…
            </p>
          )}
          {data && data.items.length === 0 && (
            <p className="text-sm text-slate-500">
              {enabled ? "Nothing saved yet. Make a prediction in the Studio." : "No saved history."}
            </p>
          )}
          {data && data.items.length > 0 && (
            <>
              <ul className="divide-y divide-slate-100 rounded-xl border border-slate-200">
                {data.items.slice(0, 20).map((item) => (
                  <li key={item.id} className="px-3.5 py-2.5">
                    <p className="tamil-text break-words text-sm font-medium text-slate-900">{item.text}</p>
                    <p className="mt-0.5 text-xs text-slate-500">
                      {item.mode === "next_word" ? "Next word" : "Auto-complete"} · {item.created_at.replace("T", " ").replace("+00:00", " UTC")}
                    </p>
                    <p className="tamil-text mt-1 text-xs text-slate-600">→ {item.suggestions.join(", ")}</p>
                  </li>
                ))}
              </ul>
              {data.items.length > 20 && (
                <p className="mt-2 text-xs text-slate-500">Showing the latest 20 of {data.items.length} saved entries.</p>
              )}
            </>
          )}
          {data && data.items.length > 0 && (
            <div className="mt-4">
              {confirming ? (
                <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Confirm clearing history">
                  <span className="text-sm text-slate-700">Delete all saved history?</span>
                  <button
                    type="button"
                    onClick={() => void clear()}
                    disabled={busy}
                    className="rounded-lg bg-red-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-red-700 disabled:opacity-60"
                  >
                    Yes, delete
                  </button>
                  <button
                    type="button"
                    onClick={() => setConfirming(false)}
                    className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => setConfirming(true)}
                  className="inline-flex items-center gap-2 rounded-lg border border-red-200 bg-white px-3.5 py-2 text-sm font-medium text-red-700 hover:bg-red-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
                >
                  <Trash2 className="h-4 w-4" aria-hidden="true" /> Clear history
                </button>
              )}
            </div>
          )}
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5" aria-labelledby="model-heading">
        <h2 id="model-heading" className="text-base font-semibold text-slate-900">
          Model and backend
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          Status: <span className="font-medium text-slate-900">{status.label}</span>
          {model ? ` — ${model.message}` : ""}
        </p>
        <button
          type="button"
          onClick={() => void refresh()}
          className="mt-3 rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
        >
          Re-check model
        </button>
        <div className="mt-4 rounded-xl bg-slate-50 p-3.5 text-sm text-slate-700">
          <p className="font-medium text-slate-900">Where the model is configured</p>
          <p className="mt-1">
            The backend reads <code className="rounded bg-white px-1 py-0.5 text-xs">MOZHI_ARTIFACT_DIR</code> (default{" "}
            <code className="rounded bg-white px-1 py-0.5 text-xs">artifacts/</code>), which must contain the model file and{" "}
            <code className="rounded bg-white px-1 py-0.5 text-xs">tokenizer.json</code> from the same training run. After changing
            files or settings, restart the backend and use &ldquo;Re-check model&rdquo;. See <code className="rounded bg-white px-1 py-0.5 text-xs">docs/setup-guide.md</code>.
          </p>
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5" aria-labelledby="privacy-heading">
        <h2 id="privacy-heading" className="mb-3 text-base font-semibold text-slate-900">
          Privacy and responsible use
        </h2>
        <PrivacyNotice />
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-slate-600">
          <li>The model is trained only on the corpus it was given. Review the corpus for personal data before training.</li>
          <li>Suggestions can reflect bias or offensive language present in the training text; do not use them blindly.</li>
          <li>Heavy reliance on predictions can push writing toward common words and away from dialects and rarer vocabulary.</li>
          <li>Requests are limited to 2,000 characters and are validated on the server.</li>
        </ul>
      </section>
    </div>
  );
}
