import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import { ModelStatusProvider } from "../hooks/ModelStatusContext";
import type { ModelStatus, PredictResponse, Suggestion } from "../types/api";

export const READY: ModelStatus = {
  state: "ready",
  ready: true,
  lstm_available: true,
  fallback_available: false,
  development_model: false,
  message: "Trained LSTM model loaded and ready.",
};

export const MISSING: ModelStatus = {
  state: "missing_artifacts",
  ready: false,
  lstm_available: false,
  fallback_available: false,
  development_model: false,
  message: "No trained model found. Train one with scripts/train_model.py.",
};

export function suggestion(rank: number, word: string, score: number): Suggestion {
  return { rank, word, score, score_type: "model_probability", source: "lstm" };
}

export function predictResponse(partial: Partial<PredictResponse> & Pick<PredictResponse, "input_text" | "mode">): PredictResponse {
  return {
    prefix: null,
    suggestions: [],
    source: "lstm",
    model_status: READY,
    message: null,
    diagnostics: { context_words: 3, unknown_context_words: 0, context_truncated: false },
    ...partial,
  };
}

export const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

type Handler = (init: RequestInit | undefined) => Response | Promise<Response>;

/** Replace global fetch with a router keyed by "METHOD /path". Unknown routes fail the test. */
export function mockApi(routes: Record<string, Handler>) {
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${String(input)}`;
    const handler = routes[key];
    if (!handler) throw new Error(`Unexpected request in test: ${key}`);
    return handler(init);
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

export function renderWithProviders(ui: ReactElement) {
  return render(
    <MemoryRouter>
      <ModelStatusProvider>{ui}</ModelStatusProvider>
    </MemoryRouter>,
  );
}

export function predictCalls(fetchMock: ReturnType<typeof mockApi>) {
  return fetchMock.mock.calls
    .filter(([url, init]) => String(url) === "/api/predict" && init?.method === "POST")
    .map(([, init]) => JSON.parse(String(init?.body)));
}
