import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, ApiError, isAbort } from "../services/api";
import type { ModelStatus } from "../types/api";

export type Connection = "checking" | "online" | "offline";

interface ModelStatusValue {
  connection: Connection;
  model: ModelStatus | null;
  error: string | null;
  refresh: () => Promise<void>;
}

const Ctx = createContext<ModelStatusValue | null>(null);

const POLL_MS = 30_000;

export function ModelStatusProvider({ children }: { children: ReactNode }) {
  const [connection, setConnection] = useState<Connection>("checking");
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    controller.current?.abort();
    const ctrl = new AbortController();
    controller.current = ctrl;
    try {
      const health = await api.health(ctrl.signal);
      setModel(health.model);
      setConnection("online");
      setError(null);
    } catch (e) {
      if (isAbort(e)) return;
      setModel(null);
      setConnection("offline");
      setError(e instanceof ApiError ? e.message : "Cannot reach the backend.");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = setInterval(() => void refresh(), POLL_MS);
    return () => {
      clearInterval(id);
      controller.current?.abort();
    };
  }, [refresh]);

  const value = useMemo(() => ({ connection, model, error, refresh }), [connection, model, error, refresh]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useModelStatus(): ModelStatusValue {
  const value = useContext(Ctx);
  if (!value) throw new Error("useModelStatus must be used inside <ModelStatusProvider>");
  return value;
}
