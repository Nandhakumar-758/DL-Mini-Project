import { ShieldCheck } from "lucide-react";

export function PrivacyNotice({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-start gap-2.5 text-xs leading-relaxed text-slate-600">
      <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-indigo-600" aria-hidden="true" />
      <p>
        <span className="font-semibold text-slate-800">Privacy: </span>
        Text you type is sent to this app&apos;s own backend to compute suggestions. It is never sent to third-party
        AI services and is not saved unless you turn on history in Settings.
        {!compact && (
          <>
            {" "}
            Suggestions reflect the training corpus and can be biased, repetitive or inappropriate; review them before
            using them.
          </>
        )}
      </p>
    </div>
  );
}
