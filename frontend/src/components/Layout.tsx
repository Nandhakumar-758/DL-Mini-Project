import { BarChart3, Keyboard, Settings, Sparkles, Workflow } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { ModelStatusBadge } from "./ModelStatusBadge";

const NAV = [
  { to: "/", label: "Prediction Studio", short: "Studio", icon: Keyboard, end: true },
  { to: "/performance", label: "Model Performance", short: "Performance", icon: BarChart3, end: false },
  { to: "/how-it-works", label: "How It Works", short: "How it works", icon: Workflow, end: false },
  { to: "/settings", label: "Settings", short: "Settings", icon: Settings, end: false },
];

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 text-white">
        <Sparkles className="h-5 w-5" aria-hidden="true" />
      </span>
      <div className="leading-tight">
        <p className="text-base font-bold tracking-tight text-slate-900">MozhiPredict AI</p>
        <p className="text-[11px] text-slate-500">Tamil · English typing assistant</p>
      </div>
    </div>
  );
}

export function Layout() {
  return (
    <div className="min-h-screen lg:flex">
      {/* Desktop sidebar */}
      <aside className="hidden w-64 shrink-0 flex-col border-r border-slate-200 bg-white px-4 py-5 lg:sticky lg:top-0 lg:flex lg:h-screen">
        <Brand />
        <nav aria-label="Main" className="mt-8 flex flex-1 flex-col gap-1">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                  isActive ? "bg-indigo-50 text-indigo-700" : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                }`
              }
            >
              <Icon className="h-4 w-4" aria-hidden="true" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-slate-200 pt-4">
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">Model</p>
          <ModelStatusBadge />
        </div>
      </aside>

      <div className="min-w-0 flex-1">
        {/* Mobile / tablet top bar */}
        <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/95 backdrop-blur lg:hidden">
          <div className="flex items-center justify-between gap-3 px-4 py-3">
            <Brand />
            <ModelStatusBadge />
          </div>
          <nav aria-label="Main" className="grid grid-cols-4 border-t border-slate-100">
            {NAV.map(({ to, short, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  `flex flex-col items-center gap-0.5 px-1 py-2 text-[11px] font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-indigo-500 ${
                    isActive ? "text-indigo-700" : "text-slate-500"
                  }`
                }
              >
                <Icon className="h-4 w-4" aria-hidden="true" />
                <span className="truncate">{short}</span>
              </NavLink>
            ))}
          </nav>
        </header>

        <main className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
