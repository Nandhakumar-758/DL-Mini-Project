# Data

Training text is **not** included, apart from a tiny illustrative sample.

## Format

- UTF-8 text files, one sentence per line, one file per language (e.g. `english.txt`, `tamil.txt`).
- Cleaning (done by the pipeline, see `backend/app/core/text.py`): English is lowercased; only
  English letters, Tamil characters (U+0B80–U+0BFF) and spaces are kept; whitespace is collapsed;
  lines with fewer than 3 words are dropped. Tamil text is never lowercased or normalised.

## Sources

The report used publicly available English text (news, Wikipedia, books) and Tamil text (Tamil
Wikipedia, news). Use only text whose licence permits this use, and record each source and licence
here when you add it:

| File | Source | Licence | Notes |
|---|---|---|---|
| _(add yours)_ | | | |

Wikipedia text is CC BY-SA; check the licence of any news or book source before using it.

## Illustrative sample

`sample/english_sample.txt` and `sample/tamil_sample.txt` are 153 short hand-written
sentences (76 English, 77 Tamil) for tests and the development model. They are **not** a public dataset and are far too
small to train a useful model.

```bash
cd backend
python -m scripts.train_model --dev      # writes ../artifacts/dev_sample
```

## Split strategy and limits

Training splits by **sentence** (80:20, seeded) so no sentence contributes to both sides, and the
tokenizer is fitted on the training part only. Near-duplicate sentences that differ slightly can
still straddle the split; remove them from the corpus if that matters for your evaluation.
