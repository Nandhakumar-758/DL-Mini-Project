import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";
import type { ReactNode } from "react";

type Kind = "info" | "warning" | "error" | "success";

const STYLES: Record<Kind, { wrap: string; icon: ReactNode }> = {
  info: { wrap: "border-indigo-200 bg-indigo-50 text-indigo-950", icon: <Info className="h-4 w-4 text-indigo-600" /> },
  warning: { wrap: "border-amber-200 bg-amber-50 text-amber-950", icon: <AlertTriangle className="h-4 w-4 text-amber-600" /> },
  error: { wrap: "border-red-200 bg-red-50 text-red-950", icon: <XCircle className="h-4 w-4 text-red-600" /> },
  success: { wrap: "border-emerald-200 bg-emerald-50 text-emerald-950", icon: <CheckCircle2 className="h-4 w-4 text-emerald-600" /> },
};

interface Props {
  kind?: Kind;
  title?: string;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}

export function Alert({ kind = "info", title, children, action, className = "" }: Props) {
  const s = STYLES[kind];
  return (
    <div
      role={kind === "error" ? "alert" : "status"}
      className={`flex items-start gap-3 rounded-xl border px-4 py-3 text-sm ${s.wrap} ${className}`}
    >
      <span className="mt-0.5 shrink-0" aria-hidden="true">
        {s.icon}
      </span>
      <div className="min-w-0 flex-1">
        {title && <p className="font-semibold">{title}</p>}
        {children && <div className={title ? "mt-0.5 opacity-90" : ""}>{children}</div>}
      </div>
      {action}
    </div>
  );
}
