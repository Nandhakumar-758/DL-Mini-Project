import { lazy, Suspense } from "react";
import { Link, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { ModelStatusProvider } from "./hooks/ModelStatusContext";
import { HowItWorks } from "./pages/HowItWorks";
import { PredictionStudio } from "./pages/PredictionStudio";
import { Settings } from "./pages/Settings";

// The charts library is large, so the performance page is loaded on demand.
const ModelPerformance = lazy(() =>
  import("./pages/ModelPerformance").then((m) => ({ default: m.ModelPerformance })),
);

function NotFound() {
  return (
    <div className="py-16 text-center">
      <h1 className="text-2xl font-bold text-slate-900">Page not found</h1>
      <p className="mt-2 text-sm text-slate-600">That page does not exist.</p>
      <Link to="/" className="mt-4 inline-block rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700">
        Back to Prediction Studio
      </Link>
    </div>
  );
}

export default function App() {
  return (
    <ModelStatusProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<PredictionStudio />} />
          <Route
            path="performance"
            element={
              <Suspense fallback={<p className="text-sm text-slate-600">Loading…</p>}>
                <ModelPerformance />
              </Suspense>
            }
          />
          <Route path="how-it-works" element={<HowItWorks />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </ModelStatusProvider>
  );
}
