import type { Mode, ScoreType } from "../types/api";

export const MAX_CHARS = 2000;

export function countWords(text: string): number {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

/** Number of user-perceived characters (grapheme clusters), so Tamil is not over-counted. */
export function countCharacters(text: string): number {
  if (typeof Intl !== "undefined" && "Segmenter" in Intl) {
    const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });
    let n = 0;
    for (const _ of segmenter.segment(text)) n += 1; // eslint-disable-line @typescript-eslint/no-unused-vars
    return n;
  }
  return Array.from(text).length;
}

export function endsWithSpace(text: string): boolean {
  return /\s$/.test(text);
}

/**
 * Insert a chosen suggestion with correct spacing.
 *
 * - autocomplete: the partially typed last word is replaced by the full word.
 * - next word: the word is appended, adding a separating space only if needed.
 * A trailing space is added so the user can keep typing (or ask for the next word).
 */
export function applySuggestion(text: string, word: string, mode: Mode): string {
  if (mode === "autocomplete" && !endsWithSpace(text) && text.length > 0) {
    return text.replace(/\S*$/, "") + word + " ";
  }
  if (text.length === 0 || endsWithSpace(text)) return text + word + " ";
  return text + " " + word + " ";
}

/** Percentage string with 2 decimals; tiny values are shown as "<0.01%". */
export function formatPercent(value: number): string {
  const pct = value * 100;
  if (pct > 0 && pct < 0.01) return "<0.01%";
  return `${pct.toFixed(2)}%`;
}

export function scoreLabel(type: ScoreType): string {
  return type === "model_probability" ? "Model probability" : "Corpus frequency";
}

export interface Example {
  label: string;
  text: string;
  mode: Mode;
}

export const EXAMPLES: Example[] = [
  { label: "i want to", text: "i want to ", mode: "next_word" },
  { label: "the weather is", text: "the weather is ", mode: "next_word" },
  { label: "நான் பள்ளிக்கு", text: "நான் பள்ளிக்கு ", mode: "next_word" },
  { label: "இன்று மழை", text: "இன்று மழை ", mode: "next_word" },
  { label: "i want to ge…", text: "i want to ge", mode: "autocomplete" },
  { label: "நான் ப…", text: "நான் ப", mode: "autocomplete" },
];
