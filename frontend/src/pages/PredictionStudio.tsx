import { Eraser, Loader2, RefreshCw, Wand2, Zap } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Alert } from "../components/Alert";
import { PrivacyNotice } from "../components/PrivacyNotice";
import { SuggestionList } from "../components/SuggestionList";
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import { useModelStatus } from "../hooks/ModelStatusContext";
import { api, isAbort } from "../services/api";
import type { Mode, PredictResponse } from "../types/api";
import {
  applySuggestion,
  countCharacters,
  countWords,
  endsWithSpace,
  EXAMPLES,
  MAX_CHARS,
  scoreLabel,
  type Example,
} from "../utils/text";

type Phase = "idle" | "loading" | "success" | "error";

const MODES: { value: Mode; label: string; hint: string }[] = [
  { value: "next_word", label: "Next-Word Prediction", hint: "Predict the word that comes next." },
  { value: "autocomplete", label: "Word Auto-Complete", hint: "Complete the word you are typing." },
];
const TOP_K_OPTIONS = [3, 5, 10];

function validateInput(text: string): string | null {
  if (!text.trim()) return "Type some Tamil or English text first.";
  if (text.length > MAX_CHARS) return `The text is too long (limit ${MAX_CHARS} characters). Shorten it and try again.`;
  return null;
}

function Segmented<T extends string | number>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-slate-200 bg-slate-100 p-0.5">
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={String(o.value)}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.value)}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
              active ? "bg-white text-indigo-700 shadow-sm" : "text-slate-600 hover:text-slate-900"
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function StatusBanner() {
  const { connection, model, error, refresh } = useModelStatus();
  if (connection === "checking") return null;
  if (connection === "offline") {
    return (
      <Alert
        kind="error"
        title="Backend offline"
        action={
          <button
            type="button"
            onClick={() => void refresh()}
            className="inline-flex items-center gap-1.5 rounded-lg border border-red-300 bg-white px-2.5 py-1 text-xs font-medium text-red-800 hover:bg-red-50"
          >
            <RefreshCw className="h-3 w-3" aria-hidden="true" /> Retry
          </button>
        }
      >
        {error}
      </Alert>
    );
  }
  if (model && !model.ready) {
    return (
      <Alert kind="warning" title="Real LSTM prediction is unavailable">
        {model.message} Until trained artifacts exist, next-word prediction cannot run
        {model.fallback_available ? ", and auto-complete will use a clearly labelled vocabulary-frequency fallback." : "."}
      </Alert>
    );
  }
  if (model?.development_model) {
    return (
      <Alert kind="info" title="Development model">
        This is a tiny model trained on the illustrative sample corpus to test the pipeline. Its suggestions are not
        representative of the full LSTM described in the report.
      </Alert>
    );
  }
  return null;
}

export function PredictionStudio() {
  const { refresh } = useModelStatus();
  const [text, setText] = useState("");
  const [mode, setMode] = useState<Mode>("next_word");
  const [topK, setTopK] = useState(5);
  const [live, setLive] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<{ code: string; message: string } | null>(null);
  const [validation, setValidation] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const editorRef = useRef<HTMLTextAreaElement | null>(null);

  const run = useCallback(
    async (value: string, m: Mode, k: number, manual = false) => {
      const problem = validateInput(value);
      if (problem) {
        if (manual) setValidation(problem);
        return;
      }
      setValidation(null);
      abortRef.current?.abort(); // only the latest request may update the screen
      const controller = new AbortController();
      abortRef.current = controller;
      setPhase("loading");
      setError(null);
      try {
        const response = await api.predict({ text: value, mode: m, top_k: k }, controller.signal);
        setResult(response);
        setPhase("success");
      } catch (e) {
        if (isAbort(e)) return;
        setResult(null);
        setError({
          code: (e as { code?: string }).code ?? "unknown",
          message: e instanceof Error ? e.message : "Something went wrong.",
        });
        setPhase("error");
        void refresh(); // the failure may mean the backend or model state changed
      }
    },
    [refresh],
  );

  // Optional live suggestions, debounced so we never fire a request per keystroke.
  const debouncedText = useDebouncedValue(text, 400);
  useEffect(() => {
    if (!live) return;
    const eligible = mode === "next_word" ? endsWithSpace(debouncedText) : !endsWithSpace(debouncedText);
    if (!debouncedText.trim() || !eligible) return;
    void run(debouncedText, mode, topK);
  }, [live, debouncedText, mode, topK, run]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const resetResults = () => {
    abortRef.current?.abort();
    setResult(null);
    setError(null);
    setPhase("idle");
  };

  const onModeChange = (m: Mode) => {
    setMode(m);
    resetResults();
  };

  const onClear = () => {
    setText("");
    setValidation(null);
    resetResults();
    editorRef.current?.focus();
  };

  const onExample = (ex: Example) => {
    setText(ex.text);
    setMode(ex.mode);
    void run(ex.text, ex.mode, topK, true);
  };

  const onSelect = (word: string) => {
    setText(applySuggestion(text, word, result?.mode ?? mode));
    resetResults();
    editorRef.current?.focus();
  };

  const chars = countCharacters(text);
  const tooLong = text.length > MAX_CHARS;
  const stale = result !== null && result.input_text !== text;
  const activeMode = MODES.find((m) => m.value === mode)!;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Prediction Studio</h1>
        <p className="mt-1 text-sm text-slate-600">
          Type Tamil, English or both, then predict the next word or complete the word you are typing.
        </p>
      </div>

      <StatusBanner />

      <div className="grid gap-5 lg:grid-cols-5">
        {/* ------------------------------------------------------------ editor */}
        <section aria-labelledby="editor-heading" className="space-y-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm lg:col-span-3 sm:p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 id="editor-heading" className="sr-only">
              Editor
            </h2>
            <Segmented
              label="Prediction mode"
              value={mode}
              onChange={onModeChange}
              options={MODES.map((m) => ({ value: m.value, label: m.label }))}
            />
            <div className="flex items-center gap-2 text-sm text-slate-600">
              <span id="topk-label">Suggestions</span>
              <Segmented
                label="Number of suggestions"
                value={topK}
                onChange={(k) => {
                  setTopK(k);
                  resetResults();
                }}
                options={TOP_K_OPTIONS.map((k) => ({ value: k, label: String(k) }))}
              />
            </div>
          </div>
          <p className="text-xs text-slate-500">{activeMode.hint}</p>

          <div>
            <label htmlFor="editor" className="sr-only">
              Text to continue
            </label>
            <textarea
              id="editor"
              ref={editorRef}
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                setValidation(null);
              }}
              onKeyDown={(e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                  e.preventDefault();
                  void run(text, mode, topK, true);
                }
              }}
              rows={8}
              spellCheck={false}
              autoComplete="off"
              placeholder={
                mode === "next_word"
                  ? "Type a sentence that ends with a space, e.g. “i want to ”  or  “நான் பள்ளிக்கு ”"
                  : "Type a partial last word, e.g. “i want to ge”  or  “நான் ப”"
              }
              aria-invalid={tooLong || validation ? true : undefined}
              aria-describedby="editor-meta"
              className="tamil-text block w-full resize-y rounded-xl border border-slate-300 bg-white px-4 py-3 text-lg text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
            />
            <div id="editor-meta" className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
              <span className={tooLong ? "font-semibold text-red-600" : ""}>
                {chars} characters · {countWords(text)} words{tooLong ? ` (limit ${MAX_CHARS})` : ""}
              </span>
              <span className="hidden sm:inline">Ctrl/⌘ + Enter to predict</span>
            </div>
            {validation && (
              <p role="alert" className="mt-2 text-sm font-medium text-red-700">
                {validation}
              </p>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <button
              type="button"
              onClick={() => void run(text, mode, topK, true)}
              disabled={phase === "loading"}
              className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-indigo-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {phase === "loading" ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              ) : (
                <Wand2 className="h-4 w-4" aria-hidden="true" />
              )}
              Predict
            </button>
            <button
              type="button"
              onClick={onClear}
              disabled={!text}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Eraser className="h-4 w-4" aria-hidden="true" /> Clear
            </button>
            <label className="ml-auto inline-flex cursor-pointer items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={live}
                onChange={(e) => setLive(e.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              <Zap className="h-3.5 w-3.5 text-slate-500" aria-hidden="true" />
              Live suggestions
            </label>
          </div>

          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Try an example</p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.label}
                  type="button"
                  onClick={() => onExample(ex)}
                  className="tamil-text rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-sm text-slate-700 transition-colors hover:border-indigo-300 hover:bg-indigo-50 hover:text-indigo-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                >
                  {ex.label}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* ------------------------------------------------------- suggestions */}
        <section
          aria-labelledby="suggestions-heading"
          aria-live="polite"
          className="space-y-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5 lg:col-span-2"
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 id="suggestions-heading" className="text-base font-semibold text-slate-900">
              Suggestions
            </h2>
            {result && result.suggestions.length > 0 && (
              <span
                className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                  result.source === "lstm" ? "bg-indigo-50 text-indigo-700" : "bg-amber-100 text-amber-900"
                }`}
              >
                {result.source === "lstm"
                  ? result.model_status.development_model
                    ? "Dev LSTM model"
                    : "LSTM model"
                  : "Vocabulary fallback · not LSTM"}
              </span>
            )}
          </div>

          {phase === "idle" && (
            <div className="rounded-xl border border-dashed border-slate-300 px-4 py-10 text-center text-sm text-slate-500">
              Ranked suggestions will appear here. Press <span className="font-semibold">Predict</span> or pick an
              example.
            </div>
          )}

          {phase === "loading" && (
            <div className="space-y-2" aria-busy="true" aria-label="Loading suggestions">
              {Array.from({ length: Math.min(topK, 5) }).map((_, i) => (
                <div key={i} className="h-[54px] animate-pulse rounded-xl bg-slate-100" />
              ))}
            </div>
          )}

          {phase === "error" && error && (
            <Alert
              kind="error"
              title={
                error.code === "backend_unreachable"
                  ? "Backend unavailable"
                  : error.code === "model_unavailable"
                    ? "LSTM model not available"
                    : "Could not get suggestions"
              }
            >
              {error.message}
            </Alert>
          )}

          {phase === "success" && result && (
            <>
              {stale && <p className="text-xs text-slate-500">The text changed after these suggestions were made.</p>}
              {result.suggestions.length > 0 ? (
                <SuggestionList suggestions={result.suggestions} onSelect={onSelect} />
              ) : (
                <Alert kind="info" title="No suggestions">
                  {result.message ?? "The model had nothing to suggest for this input."}
                </Alert>
              )}
              {result.suggestions.length > 0 && result.message && (
                <Alert kind={result.source === "vocabulary_fallback" ? "warning" : "info"}>{result.message}</Alert>
              )}
              {result.diagnostics.unknown_context_words > 0 && result.diagnostics.context_words > 0 && (
                <p className="text-xs text-slate-500">
                  {result.diagnostics.unknown_context_words} of {result.diagnostics.context_words} typed words are
                  outside the model vocabulary (treated as &lt;OOV&gt;), which lowers quality.
                </p>
              )}
              {result.diagnostics.context_truncated && (
                <p className="text-xs text-slate-500">Only the last 10 words are used as context.</p>
              )}
              {result.suggestions.length > 0 && (
                <p className="border-t border-slate-100 pt-3 text-xs leading-relaxed text-slate-500">
                  {scoreLabel(result.suggestions[0].score_type)}: the share of probability the model gives each word
                  over its <em>entire</em> vocabulary. Values do not add up to 100% and are not confidence levels. Bar
                  length is relative to the top suggestion. Click a word to insert it.
                </p>
              )}
            </>
          )}
        </section>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white px-4 py-3">
        <PrivacyNotice />
      </div>
    </div>
  );
}
