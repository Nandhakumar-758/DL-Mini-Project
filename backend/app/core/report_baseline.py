"""Results quoted from the project report (sections 5.1, 5.5, 5.9).

These are *historical, report-reported* numbers from the authors' Colab run.
They are NOT measured by the model that is currently loaded and the UI always
labels them as a baseline. Only the per-epoch log (5.5) and the evaluation
tables (5.9) are used: the report's accuracy/loss *figures* (5.6, 5.7) show a
20-epoch run that is inconsistent with the log (early stopping after epoch 6),
so they are deliberately not reproduced.
"""

REPORT_BASELINE = {
    "label": "Report-reported baseline (Assignment 02 report, Google Colab run)",
    "note": (
        "Quoted from the report, not measured by the model currently loaded. "
        "The report fits the tokenizer on the whole corpus and splits individual "
        "n-gram pairs at random, so these numbers are optimistic compared with "
        "this project's sentence-level split."
    ),
    "corpus": {
        "total_sentences": 11996,
        "tamil_sentences": 6199,
        "english_sentences": 5797,
        "total_words": 178367,
        "unique_words": 40912,
        "vocabulary_size": 10000,
        "training_pairs": 166371,
    },
    "training": {"epochs_run": 6, "best_epoch": 3, "stopped_early": True},
    "history": {
        "loss": [6.7248, 6.1337, 5.7798, 5.5799, 5.4288, 5.3010],
        "accuracy": [0.2009, 0.2076, 0.2162, 0.2249, 0.2292, 0.2326],
        "top5_accuracy": [0.2962, 0.3017, 0.3149, 0.3257, 0.3317, 0.3371],
        "val_loss": [6.5916, 6.2171, 6.2001, 6.3146, 6.3645, 6.5378],
        "val_accuracy": [0.2060, 0.2105, 0.2208, 0.2243, 0.2256, 0.2283],
        "val_top5_accuracy": [0.2952, 0.3002, 0.3183, 0.3223, 0.3263, 0.3305],
    },
    "evaluation": {
        "accuracy": 0.2208,
        "top5_accuracy": 0.3183,
        "loss": 6.2001,
        "perplexity": 492.78,
        "oov_target_share": 0.1998,
    },
    "known_word": [
        {"language": "English", "samples": 17973, "top1": 0.1037, "top5": 0.2242, "oov_targets": 0.110},
        {"language": "Tamil", "samples": 8652, "top1": 0.0084, "top5": 0.0353, "oov_targets": 0.338},
        {"language": "All", "samples": 26625, "top1": 0.0728, "top5": 0.1628, "oov_targets": 0.200},
    ],
}
