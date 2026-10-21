import type { Suggestion } from "../types/api";
import { formatPercent, scoreLabel } from "../utils/text";

interface Props {
  suggestions: Suggestion[];
  onSelect: (word: string) => void;
}

/**
 * Ranked suggestions. Bar length is relative to the best suggestion shown (so small
 * probabilities stay visible); the printed percentage is the real score.
 */
export function SuggestionList({ suggestions, onSelect }: Props) {
  const max = Math.max(...suggestions.map((s) => s.score), Number.EPSILON);
  return (
    <ol className="space-y-2" aria-label="Ranked suggestions">
      {suggestions.map((s) => (
        <li key={`${s.rank}-${s.word}`}>
          <button
            type="button"
            onClick={() => onSelect(s.word)}
            aria-label={`Insert ${s.word}, rank ${s.rank}, ${scoreLabel(s.score_type)} ${formatPercent(s.score)}`}
            className="group w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-left transition-colors hover:border-indigo-300 hover:bg-indigo-50/50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            <div className="flex items-center gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-slate-100 text-xs font-semibold text-slate-600 group-hover:bg-indigo-100 group-hover:text-indigo-700">
                {s.rank}
              </span>
              <span className="tamil-text min-w-0 flex-1 truncate text-base font-medium text-slate-900">{s.word}</span>
              <span className="shrink-0 font-mono text-sm tabular-nums text-slate-700">{formatPercent(s.score)}</span>
            </div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100" aria-hidden="true">
              <div
                className={`h-full rounded-full transition-[width] duration-300 ${
                  s.source === "lstm" ? "bg-indigo-500" : "bg-amber-400"
                }`}
                style={{ width: `${Math.max(2, (s.score / max) * 100)}%` }}
              />
            </div>
          </button>
        </li>
      ))}
    </ol>
  );
}
