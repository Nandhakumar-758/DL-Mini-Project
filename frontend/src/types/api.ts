// Mirrors backend/app/schemas/api.py

export type Mode = "next_word" | "autocomplete";
export type ScoreType = "model_probability" | "vocabulary_frequency";
export type Source = "lstm" | "vocabulary_fallback";

export interface ModelStatus {
  state: string;
  ready: boolean;
  lstm_available: boolean;
  fallback_available: boolean;
  development_model: boolean;
  message: string;
}

export interface Suggestion {
  word: string;
  rank: number;
  score: number;
  score_type: ScoreType;
  source: Source;
}

export interface Diagnostics {
  context_words: number;
  unknown_context_words: number;
  context_truncated: boolean;
}

export interface PredictRequest {
  text: string;
  mode: Mode;
  top_k: number;
}

export interface PredictResponse {
  input_text: string;
  mode: Mode;
  prefix: string | null;
  suggestions: Suggestion[];
  source: Source | null;
  model_status: ModelStatus;
  message: string | null;
  diagnostics: Diagnostics;
}

export interface HealthResponse {
  status: "ok";
  model: ModelStatus;
}

export interface HistoryItem {
  id: number;
  created_at: string;
  text: string;
  mode: Mode;
  top_k: number;
  suggestions: string[];
}

export interface HistoryResponse {
  enabled: boolean;
  items: HistoryItem[];
}

export interface LayerInfo {
  type: string;
  output_shape: (number | null)[] | null;
  parameters: number;
  units?: number;
  rate?: number;
  output_dim?: number;
  activation?: string;
}

export interface ModelInfo {
  status: ModelStatus;
  architecture: {
    sequence_length: number;
    vocab_size: number;
    total_parameters: number;
    layers: LayerInfo[];
  } | null;
  tokenizer: {
    vocab_size: number;
    num_words: number;
    oov_token: string | null;
    distinct_words_seen: number;
    has_frequency_table: boolean;
  } | null;
  run: { run_id?: string; trained_at?: string; mode?: string; libraries?: Record<string, string> } | null;
  config: Record<string, number | string | boolean | null> | null;
  dataset: Record<string, number | string | boolean | string[] | null> | null;
  training: {
    epochs_run?: number;
    best_epoch?: number;
    stopped_early?: boolean;
    restored_best_weights?: boolean;
    seconds?: number;
  } | null;
}

export type TrainingHistory = Record<string, number[]>;

export interface KnownWordBucket {
  samples: number;
  known_word_samples: number;
  known_word_top1_accuracy: number | null;
  known_word_top5_accuracy: number | null;
  oov_target_share: number | null;
}

export interface Evaluation {
  evaluated_at?: string;
  split_strategy?: string;
  samples: number;
  loss: number;
  accuracy: number;
  top5_accuracy: number;
  perplexity: number;
  oov_target_share: number;
  known_word: Record<string, KnownWordBucket>;
}

export interface ReportBaseline {
  label: string;
  note: string;
  corpus: Record<string, number>;
  training: { epochs_run: number; best_epoch: number; stopped_early: boolean };
  history: TrainingHistory;
  evaluation: { accuracy: number; top5_accuracy: number; loss: number; perplexity: number; oov_target_share: number };
  known_word: { language: string; samples: number; top1: number; top5: number; oov_targets: number }[];
}

export interface MetricsResponse {
  available: boolean;
  message: string;
  development_model: boolean;
  history: TrainingHistory | null;
  evaluation: Evaluation | null;
  training: ModelInfo["training"];
  dataset: ModelInfo["dataset"];
  report_baseline: ReportBaseline;
}
