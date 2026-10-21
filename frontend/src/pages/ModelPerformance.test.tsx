import { screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { json, MISSING, mockApi, renderWithProviders } from "../test/helpers";
import { ModelPerformance } from "./ModelPerformance";

afterEach(() => vi.unstubAllGlobals());

const baseline = {
  label: "Report-reported baseline (Assignment 02 report, Google Colab run)",
  note: "Quoted from the report, not measured by the model currently loaded.",
  corpus: { total_sentences: 11996, tamil_sentences: 6199, english_sentences: 5797, total_words: 178367, unique_words: 40912, vocabulary_size: 10000, training_pairs: 166371 },
  training: { epochs_run: 6, best_epoch: 3, stopped_early: true },
  history: { loss: [], accuracy: [], top5_accuracy: [], val_loss: [], val_accuracy: [], val_top5_accuracy: [] },
  evaluation: { accuracy: 0.2208, top5_accuracy: 0.3183, loss: 6.2001, perplexity: 492.78, oov_target_share: 0.1998 },
  known_word: [{ language: "English", samples: 17973, top1: 0.1037, top5: 0.2242, oov_targets: 0.11 }],
};

describe("Model Performance", () => {
  it("marks metrics as not available and labels the report values as a baseline", async () => {
    mockApi({
      "GET /api/health": () => json({ status: "ok", model: MISSING }),
      "GET /api/metrics": () =>
        json({
          available: false,
          message: "Metrics not available. Run scripts/train_model.py to generate history.json and evaluation.json.",
          development_model: false,
          history: null,
          evaluation: null,
          training: null,
          dataset: null,
          report_baseline: baseline,
        }),
      "GET /api/model-info": () => json({ status: MISSING, architecture: null, tokenizer: null, run: null, config: null, dataset: null, training: null }),
    });
    renderWithProviders(<ModelPerformance />);
    expect(await screen.findByText("Metrics not available")).toBeInTheDocument();
    expect(screen.queryByText("Measured validation results")).not.toBeInTheDocument();
    expect(screen.getByText(/Report-reported baseline/)).toBeInTheDocument();
    expect(screen.getAllByText(/not measured by the model/i).length).toBeGreaterThan(0);
    expect(screen.getByText("492.78")).toBeInTheDocument();
  });

  it("shows an error when the backend is down", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    renderWithProviders(<ModelPerformance />);
    expect(await screen.findByText("Could not load metrics")).toBeInTheDocument();
  });
});
