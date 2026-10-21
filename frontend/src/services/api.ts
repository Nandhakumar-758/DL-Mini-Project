import type {
  HealthResponse,
  HistoryResponse,
  MetricsResponse,
  ModelInfo,
  PredictRequest,
  PredictResponse,
} from "../types/api";

const BASE: string = import.meta.env.VITE_API_BASE_URL ?? "";

/** An error with a machine-readable code and a message that is safe to show to the user. */
export class ApiError extends Error {
  code: string;
  status?: number;

  constructor(code: string, message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
    });
  } catch (error) {
    if (isAbort(error)) throw error;
    throw new ApiError(
      "backend_unreachable",
      "Cannot reach the MozhiPredict backend. Start it with: uvicorn app.main:app --port 8000",
    );
  }

  if (!response.ok) {
    let code = "http_error";
    let message = `The server returned an error (HTTP ${response.status}).`;
    try {
      const body = await response.json();
      const detail = body?.detail;
      if (detail && typeof detail === "object") {
        code = detail.code ?? code;
        message = detail.message ?? message;
      } else if (typeof detail === "string") {
        message = detail;
      }
    } catch {
      /* non-JSON error body: keep the generic message */
    }
    throw new ApiError(code, message, response.status);
  }

  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError("invalid_response", "The server sent a response that could not be read.");
  }
}

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>("/api/health", { signal }),
  modelInfo: (signal?: AbortSignal) => request<ModelInfo>("/api/model-info", { signal }),
  metrics: (signal?: AbortSignal) => request<MetricsResponse>("/api/metrics", { signal }),
  predict: (body: PredictRequest, signal?: AbortSignal) =>
    request<PredictResponse>("/api/predict", { method: "POST", body: JSON.stringify(body), signal }),
  history: () => request<HistoryResponse>("/api/history"),
  clearHistory: () => request<{ deleted: number }>("/api/history", { method: "DELETE" }),
  setHistoryEnabled: (enabled: boolean) =>
    request<{ history_enabled: boolean }>("/api/settings", {
      method: "PUT",
      body: JSON.stringify({ history_enabled: enabled }),
    }),
};

export { isAbort };
