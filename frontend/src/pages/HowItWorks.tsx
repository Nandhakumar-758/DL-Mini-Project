import {
  ArrowRight,
  Binary,
  BookOpenText,
  Brain,
  Database,
  Eraser,
  ListOrdered,
  Network,
  Percent,
  Rows3,
  type LucideIcon,
} from "lucide-react";
import type { ReactNode } from "react";

interface Step {
  title: string;
  detail: string;
  icon: LucideIcon;
}

const PIPELINE: Step[] = [
  { title: "Text Corpus", detail: "Tamil and English sentences, one per line, UTF-8.", icon: Database },
  { title: "Text Preprocessing", detail: "Lower-case English, keep only English letters, Tamil characters and spaces.", icon: Eraser },
  { title: "Tokenization", detail: "Each word becomes an integer id; the 10,000 most frequent words are kept.", icon: Binary },
  { title: "Sequence Generation", detail: "Every sentence yields (previous words → next word) training pairs.", icon: Rows3 },
  { title: "Word Embedding", detail: "Each id maps to a learned 128-number vector.", icon: Network },
  { title: "Stacked LSTM", detail: "Two LSTM layers (256 units, dropout 0.3) read the 10-word context.", icon: Brain },
  { title: "Softmax Probabilities", detail: "A probability for every word in the vocabulary.", icon: Percent },
  { title: "Ranked Suggestions", detail: "Top-k words, with padding and <OOV> removed; auto-complete keeps prefix matches.", icon: ListOrdered },
];

const CONCEPTS: { title: string; body: ReactNode }[] = [
  {
    title: "Why an LSTM?",
    body: "Language depends on earlier words. A plain recurrent network tends to forget them (the vanishing-gradient problem). An LSTM keeps a memory cell controlled by forget, input and output gates, so it can carry useful context forward and drop what is irrelevant.",
  },
  {
    title: "What do word embeddings do?",
    body: "An integer id has no meaning by itself. The embedding layer turns each id into a vector that is learned together with the LSTM, so words used in similar contexts end up with similar vectors, in both Tamil and English.",
  },
  {
    title: "How does next-word prediction work?",
    body: "The last 10 words (left-padded with zeros if shorter) go through the network. The final softmax layer outputs one probability per vocabulary word, and the highest-ranked words are shown. It is a classification problem with as many classes as there are words.",
  },
  {
    title: "Top-1 and Top-5 accuracy",
    body: "Top-1 asks whether the single most probable word is the real next word. Top-5 asks whether the real word is among the five most probable. Many words can follow the same context, so Top-5 is the fairer measure for a suggestion bar.",
  },
  {
    title: "What does perplexity measure?",
    body: "Perplexity is exp(validation loss). Roughly, it is how many equally likely words the model is torn between at each step. Lower is better: the model is less “surprised” by real text.",
  },
  {
    title: "Why is Tamil harder?",
    body: "Tamil is agglutinative: one root takes many suffixes for case, tense, number and person. That creates a huge number of distinct word forms, so the training data is spread thinly and many forms never reach the 10,000-word vocabulary.",
  },
  {
    title: "Why do out-of-vocabulary words hurt?",
    body: "Words outside the vocabulary are replaced by a single <OOV> token. The model cannot suggest them, and as context they all look identical. In the report about 20% of validation targets were <OOV> (34% for Tamil), which caps achievable accuracy.",
  },
];

const ARCH: [string, string, string][] = [
  ["Input sequence", "(10)", "0"],
  ["Embedding", "(10, 128)", "1,280,000"],
  ["LSTM 1 · 256 units · return sequences", "(10, 256)", "394,240"],
  ["Dropout 0.3", "(10, 256)", "0"],
  ["LSTM 2 · 256 units", "(256)", "525,312"],
  ["Dropout 0.3", "(256)", "0"],
  ["Dense · 256 units · ReLU", "(256)", "65,792"],
  ["Dense · softmax", "(10,000)", "2,570,000"],
];

function Chip({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "indigo" | "amber" }) {
  const tones = {
    slate: "border-slate-200 bg-slate-50 text-slate-700",
    indigo: "border-indigo-200 bg-indigo-50 text-indigo-800",
    amber: "border-amber-200 bg-amber-50 text-amber-900",
  };
  return <span className={`tamil-text rounded-md border px-2 py-1 font-mono text-xs ${tones[tone]}`}>{children}</span>;
}

export function HowItWorks() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">How It Works</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-600">
          MozhiPredict AI is a word-level language model. It reads the words you have typed and estimates which word is
          most likely to come next, for Tamil, English and mixed text.
        </p>
      </div>

      <section aria-labelledby="pipeline-heading" className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
        <h2 id="pipeline-heading" className="text-base font-semibold text-slate-900">
          From corpus to suggestions
        </h2>
        <ol className="mt-5 grid gap-x-8 gap-y-5 sm:grid-cols-2 lg:grid-cols-4">
          {PIPELINE.map(({ title, detail, icon: Icon }, i) => (
            <li key={title} className="relative rounded-xl border border-slate-200 bg-slate-50/60 p-4">
              <div className="flex items-center gap-2.5">
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white">
                  <Icon className="h-4 w-4" aria-hidden="true" />
                </span>
                <span className="text-xs font-semibold text-indigo-700">Step {i + 1}</span>
              </div>
              <p className="mt-3 text-sm font-semibold text-slate-900">{title}</p>
              <p className="mt-1 text-xs leading-relaxed text-slate-600">{detail}</p>
              {i < PIPELINE.length - 1 && (
                <ArrowRight
                  className={`absolute -right-[22px] top-1/2 hidden h-4 w-4 -translate-y-1/2 text-slate-400 ${
                    i % 4 === 3 ? "" : "lg:block"
                  } ${i % 2 === 1 ? "" : "sm:max-lg:block"}`}
                  aria-hidden="true"
                />
              )}
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="example-heading" className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
        <h2 id="example-heading" className="text-base font-semibold text-slate-900">
          One prediction, step by step
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          For the input <span className="font-mono">“i want to ”</span> the model always receives exactly 10 word slots. Unused slots at the
          start are filled with padding (id 0).
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-1.5" aria-label="Ten input slots">
          {Array.from({ length: 7 }).map((_, i) => (
            <Chip key={i}>PAD</Chip>
          ))}
          <Chip tone="indigo">i</Chip>
          <Chip tone="indigo">want</Chip>
          <Chip tone="indigo">to</Chip>
          <ArrowRight className="mx-1 h-4 w-4 text-slate-400" aria-hidden="true" />
          <Chip tone="amber">Embedding → LSTM → LSTM → Dense → Softmax</Chip>
          <ArrowRight className="mx-1 h-4 w-4 text-slate-400" aria-hidden="true" />
          <Chip tone="indigo">probability for every word</Chip>
        </div>
        <p className="mt-3 text-sm text-slate-600">
          <span className="font-semibold text-slate-800">Auto-complete</span> uses the same probabilities: for{" "}
          <span className="font-mono">“i want to ge”</span> the context is <span className="font-mono">“i want to”</span>, and only
          vocabulary words that start with <span className="font-mono">ge</span> are kept, ranked by the model&apos;s probability.
          Tamil prefixes are matched code point by code point, so typing <span className="tamil-text font-mono">க</span> also
          matches words starting with <span className="tamil-text font-mono">கி</span> or <span className="tamil-text font-mono">கு</span>.
        </p>
      </section>

      <section aria-labelledby="concepts-heading">
        <h2 id="concepts-heading" className="mb-3 flex items-center gap-2 text-base font-semibold text-slate-900">
          <BookOpenText className="h-4 w-4 text-indigo-600" aria-hidden="true" /> Key ideas
        </h2>
        <div className="grid gap-4 md:grid-cols-2">
          {CONCEPTS.map((c) => (
            <article key={c.title} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
              <h3 className="text-sm font-semibold text-slate-900">{c.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-600">{c.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section aria-labelledby="arch-heading" className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
        <h2 id="arch-heading" className="text-base font-semibold text-slate-900">
          Architecture described in the report
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          Shapes and parameter counts for a 10,000-word vocabulary, as printed in the report (4,835,344 parameters in total).
          The model currently loaded may be smaller; see Model Performance for what is actually loaded.
        </p>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[420px] text-left text-sm">
            <thead className="text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="py-2 pr-3 font-medium">Layer</th>
                <th className="py-2 pr-3 font-medium">Output shape</th>
                <th className="py-2 text-right font-medium">Parameters</th>
              </tr>
            </thead>
            <tbody>
              {ARCH.map(([layer, shape, params]) => (
                <tr key={layer} className="border-t border-slate-100">
                  <td className="py-2 pr-3 font-medium text-slate-800">{layer}</td>
                  <td className="py-2 pr-3 font-mono text-xs text-slate-600">{shape}</td>
                  <td className="py-2 text-right tabular-nums text-slate-700">{params}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section aria-labelledby="limits-heading" className="rounded-2xl border border-amber-200 bg-amber-50/60 p-4 sm:p-6">
        <h2 id="limits-heading" className="text-base font-semibold text-amber-950">
          Limitations to keep in mind
        </h2>
        <ul className="mt-2 list-disc space-y-1.5 pl-5 text-sm text-amber-950/90">
          <li>Suggestion quality depends on the size, domain and cleanliness of the training corpus.</li>
          <li>The model reproduces patterns, and biases or offensive phrases, found in its training text.</li>
          <li>Predictions can be repetitive and may nudge writing toward the most common words.</li>
          <li>Mixed Tamil-English sentences work only as far as such examples appear in the corpus.</li>
          <li>Probabilities are relative scores over the vocabulary, not a guarantee that a suggestion is correct.</li>
        </ul>
      </section>
    </div>
  );
}
