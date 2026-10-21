import { describe, expect, it } from "vitest";
import { applySuggestion, countCharacters, countWords, formatPercent } from "./text";

describe("applySuggestion", () => {
  it("appends after a trailing space (next-word)", () => {
    expect(applySuggestion("i want to ", "go", "next_word")).toBe("i want to go ");
  });

  it("adds a separating space when the text does not end with one (next-word)", () => {
    expect(applySuggestion("i want to", "go", "next_word")).toBe("i want to go ");
  });

  it("works on empty text", () => {
    expect(applySuggestion("", "hello", "next_word")).toBe("hello ");
  });

  it("replaces the partial word in auto-complete mode", () => {
    expect(applySuggestion("i want to ge", "get", "autocomplete")).toBe("i want to get ");
    expect(applySuggestion("ge", "get", "autocomplete")).toBe("get ");
  });

  it("keeps Tamil text intact and completes a Tamil prefix", () => {
    expect(applySuggestion("நான் ப", "பள்ளிக்கு", "autocomplete")).toBe("நான் பள்ளிக்கு ");
    expect(applySuggestion("நான் ", "பள்ளிக்கு", "next_word")).toBe("நான் பள்ளிக்கு ");
  });

  it("preserves line breaks before the completed word", () => {
    expect(applySuggestion("first line\nsec", "second", "autocomplete")).toBe("first line\nsecond ");
  });
});

describe("counting", () => {
  it("counts words across scripts", () => {
    expect(countWords("")).toBe(0);
    expect(countWords("  i   want\nto ")).toBe(3);
    expect(countWords("நான் பள்ளிக்கு செல்கிறேன்")).toBe(3);
  });

  it("counts Tamil by visible characters, not code points", () => {
    expect(countCharacters("நி")).toBe(1); // consonant + vowel sign = one grapheme
    expect(countCharacters("abc")).toBe(3);
  });
});

describe("formatPercent", () => {
  it("formats probabilities", () => {
    expect(formatPercent(0.1378)).toBe("13.78%");
    expect(formatPercent(0)).toBe("0.00%");
    expect(formatPercent(0.00001)).toBe("<0.01%");
  });
});
