import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ModelStatusBadge } from "../components/ModelStatusBadge";
import { json, MISSING, mockApi, predictCalls, predictResponse, READY, renderWithProviders, suggestion } from "../test/helpers";
import { PredictionStudio } from "./PredictionStudio";

afterEach(() => vi.unstubAllGlobals());

const healthOk = () => json({ status: "ok", model: READY });

async function typeInEditor(text: string) {
  const user = userEvent.setup();
  const editor = screen.getByLabelText("Text to continue");
  await user.click(editor);
  await user.paste(text); // also exercises Tamil input
  return { user, editor: editor as HTMLTextAreaElement };
}

describe("model status badge", () => {
  it("shows a loading state and then the ready state", async () => {
    mockApi({ "GET /api/health": healthOk });
    renderWithProviders(<ModelStatusBadge />);
    expect(screen.getByRole("status")).toHaveTextContent("Checking model");
    expect(await screen.findByText("LSTM ready")).toBeInTheDocument();
  });

  it("shows that no model is loaded", async () => {
    mockApi({ "GET /api/health": () => json({ status: "ok", model: MISSING }) });
    renderWithProviders(<ModelStatusBadge />);
    expect(await screen.findByText("Model not loaded")).toBeInTheDocument();
  });

  it("shows when the backend cannot be reached", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    renderWithProviders(<ModelStatusBadge />);
    expect(await screen.findByText("Backend offline")).toBeInTheDocument();
  });
});

describe("Prediction Studio", () => {
  it("sends the text, mode and top-k and renders ranked suggestions with probabilities", async () => {
    const fetchMock = mockApi({
      "GET /api/health": healthOk,
      "POST /api/predict": () =>
        json(
          predictResponse({
            input_text: "i want to ",
            mode: "next_word",
            suggestions: [suggestion(1, "go", 0.3125), suggestion(2, "eat", 0.1), suggestion(3, "see", 0.0042)],
          }),
        ),
    });
    renderWithProviders(<PredictionStudio />);
    const { user } = await typeInEditor("i want to ");
    await user.click(screen.getByRole("button", { name: /predict/i }));

    const list = await screen.findByRole("list", { name: "Ranked suggestions" });
    const items = within(list).getAllByRole("listitem");
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveTextContent("go");
    expect(items[0]).toHaveTextContent("31.25%");
    expect(items[2]).toHaveTextContent("0.42%");
    expect(predictCalls(fetchMock)).toEqual([{ text: "i want to ", mode: "next_word", top_k: 5 }]);
    expect(screen.getByText("LSTM model")).toBeInTheDocument();
    expect(screen.getByText(/do not add up to 100%/i)).toBeInTheDocument();
  });

  it("inserts the clicked suggestion into the editor with correct spacing", async () => {
    mockApi({
      "GET /api/health": healthOk,
      "POST /api/predict": () =>
        json(predictResponse({ input_text: "i want to", mode: "next_word", suggestions: [suggestion(1, "go", 0.4)] })),
    });
    renderWithProviders(<PredictionStudio />);
    const { user, editor } = await typeInEditor("i want to");
    await user.click(screen.getByRole("button", { name: /predict/i }));
    await user.click(await screen.findByRole("button", { name: /insert go/i }));
    expect(editor.value).toBe("i want to go ");
    expect(screen.queryByRole("list", { name: "Ranked suggestions" })).not.toBeInTheDocument();
  });

  it("completes the partial word in auto-complete mode", async () => {
    const fetchMock = mockApi({
      "GET /api/health": healthOk,
      "POST /api/predict": () =>
        json(
          predictResponse({
            input_text: "i want to ge",
            mode: "autocomplete",
            prefix: "ge",
            suggestions: [suggestion(1, "get", 0.0019), suggestion(2, "general", 0.0007)],
          }),
        ),
    });
    renderWithProviders(<PredictionStudio />);
    const { user, editor } = await typeInEditor("i want to ge");
    await user.click(screen.getByRole("radio", { name: "Word Auto-Complete" }));
    await user.click(screen.getByRole("button", { name: /predict/i }));
    await user.click(await screen.findByRole("button", { name: /insert get/i }));
    expect(editor.value).toBe("i want to get ");
    expect(predictCalls(fetchMock)[0]).toMatchObject({ mode: "autocomplete", text: "i want to ge" });
  });

  it("sends Tamil text unchanged and shows Tamil suggestions", async () => {
    const fetchMock = mockApi({
      "GET /api/health": healthOk,
      "POST /api/predict": () =>
        json(predictResponse({ input_text: "நான் ", mode: "next_word", suggestions: [suggestion(1, "பள்ளிக்கு", 0.12)] })),
    });
    renderWithProviders(<PredictionStudio />);
    const { user, editor } = await typeInEditor("நான் ");
    await user.click(screen.getByRole("button", { name: /predict/i }));
    await user.click(await screen.findByRole("button", { name: /insert பள்ளிக்கு/i }));
    expect(predictCalls(fetchMock)[0].text).toBe("நான் ");
    expect(editor.value).toBe("நான் பள்ளிக்கு ");
  });

  it("respects the top-k selector", async () => {
    const fetchMock = mockApi({
      "GET /api/health": healthOk,
      "POST /api/predict": () => json(predictResponse({ input_text: "hello ", mode: "next_word", suggestions: [suggestion(1, "world", 0.2)] })),
    });
    renderWithProviders(<PredictionStudio />);
    const { user } = await typeInEditor("hello ");
    await user.click(screen.getByRole("radio", { name: "10" }));
    await user.click(screen.getByRole("button", { name: /predict/i }));
    await screen.findByRole("list", { name: "Ranked suggestions" });
    expect(predictCalls(fetchMock)[0].top_k).toBe(10);
  });

  it("validates empty input without calling the backend", async () => {
    const fetchMock = mockApi({ "GET /api/health": healthOk });
    renderWithProviders(<PredictionStudio />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /predict/i }));
    expect(await screen.findByText(/type some tamil or english text first/i)).toBeInTheDocument();
    expect(predictCalls(fetchMock)).toHaveLength(0);
  });

  it("explains when the LSTM model is not available (HTTP 503)", async () => {
    mockApi({
      "GET /api/health": () => json({ status: "ok", model: MISSING }),
      "POST /api/predict": () =>
        json({ detail: { code: "model_unavailable", message: "No trained model found. Train one with scripts/train_model.py." } }, 503),
    });
    renderWithProviders(<PredictionStudio />);
    expect(await screen.findByText("Real LSTM prediction is unavailable")).toBeInTheDocument();
    const { user } = await typeInEditor("i want to ");
    await user.click(screen.getByRole("button", { name: /predict/i }));
    expect(await screen.findByText("LSTM model not available")).toBeInTheDocument();
    expect(screen.getAllByText(/Train one with scripts\/train_model.py/).length).toBeGreaterThan(0);
  });

  it("shows a clear error when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    renderWithProviders(<PredictionStudio />);
    const { user } = await typeInEditor("i want to ");
    await user.click(screen.getByRole("button", { name: /predict/i }));
    expect(await screen.findByText("Backend unavailable")).toBeInTheDocument();
    expect(screen.getAllByText(/Cannot reach the MozhiPredict backend/).length).toBeGreaterThan(0);
  });

  it("labels fallback suggestions as not coming from the LSTM", async () => {
    mockApi({
      "GET /api/health": () => json({ status: "ok", model: { ...MISSING, fallback_available: true } }),
      "POST /api/predict": () =>
        json(
          predictResponse({
            input_text: "i want to g",
            mode: "autocomplete",
            source: "vocabulary_fallback",
            suggestions: [{ rank: 1, word: "go", score: 0.01, score_type: "vocabulary_frequency", source: "vocabulary_fallback" }],
            message: "Vocabulary-frequency fallback (not an LSTM prediction): the trained LSTM is not loaded.",
          }),
        ),
    });
    renderWithProviders(<PredictionStudio />);
    const { user } = await typeInEditor("i want to g");
    await user.click(screen.getByRole("radio", { name: "Word Auto-Complete" }));
    await user.click(screen.getByRole("button", { name: /predict/i }));
    expect(await screen.findByText("Vocabulary fallback · not LSTM")).toBeInTheDocument();
    expect(screen.getByText(/not an LSTM prediction/)).toBeInTheDocument();
  });

  it("shows the backend message when there are no suggestions", async () => {
    mockApi({
      "GET /api/health": healthOk,
      "POST /api/predict": () =>
        json(predictResponse({ input_text: "i want qzx", mode: "autocomplete", message: "No vocabulary words start with “qzx”." })),
    });
    renderWithProviders(<PredictionStudio />);
    const { user } = await typeInEditor("i want qzx");
    await user.click(screen.getByRole("radio", { name: "Word Auto-Complete" }));
    await user.click(screen.getByRole("button", { name: /predict/i }));
    expect(await screen.findByText("No suggestions")).toBeInTheDocument();
    expect(screen.getByText(/No vocabulary words start with/)).toBeInTheDocument();
  });

  it("clears the text", async () => {
    mockApi({ "GET /api/health": healthOk });
    renderWithProviders(<PredictionStudio />);
    const { user, editor } = await typeInEditor("hello world");
    expect(screen.getByText(/11 characters · 2 words/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /clear/i }));
    expect(editor.value).toBe("");
  });

  it("debounces live suggestions instead of calling the API per keystroke", async () => {
    const fetchMock = mockApi({
      "GET /api/health": healthOk,
      "POST /api/predict": () => json(predictResponse({ input_text: "i want to ", mode: "next_word", suggestions: [suggestion(1, "go", 0.3)] })),
    });
    renderWithProviders(<PredictionStudio />);
    const user = userEvent.setup();
    await user.click(screen.getByLabelText(/live suggestions/i));
    await user.click(screen.getByLabelText("Text to continue"));
    await user.keyboard("i want to ");
    await waitFor(() => expect(predictCalls(fetchMock).length).toBeGreaterThan(0), { timeout: 3000 });
    await new Promise((r) => setTimeout(r, 700));
    expect(predictCalls(fetchMock)).toHaveLength(1); // 10 keystrokes, one request
  });

  it("runs an example immediately", async () => {
    const fetchMock = mockApi({
      "GET /api/health": healthOk,
      "POST /api/predict": () => json(predictResponse({ input_text: "நான் பள்ளிக்கு ", mode: "next_word", suggestions: [suggestion(1, "செல்கிறேன்", 0.5)] })),
    });
    renderWithProviders(<PredictionStudio />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "நான் பள்ளிக்கு" }));
    await screen.findByRole("list", { name: "Ranked suggestions" });
    expect(predictCalls(fetchMock)[0]).toEqual({ text: "நான் பள்ளிக்கு ", mode: "next_word", top_k: 5 });
    expect((screen.getByLabelText("Text to continue") as HTMLTextAreaElement).value).toBe("நான் பள்ளிக்கு ");
  });
});
