# BiasDetect-Corpus v1
Built 2026-10-28 with Python 3.12.3, pandas 3.0.6, scikit-learn 1.9.1.
- biasdetect_corpus.csv: 432 texts (24 prompts x 6 models x 3 samples) with round-1 labels of annotators A1-A3 (1/0/?),
  gold_raw = round-1 majority vote (PRIMARY), gold_reviewed = after the second round on 44 items (sensitivity only),
  refusal flag (regex, <25 words), is_core (GENDER/ORIGIN/RELIGION).
- agreement_summary_round1.csv: Fleiss kappa (prompt-level bootstrap CI), Gwet AC1, pairwise Cohen kappa, on items with 3 valid labels.
- Annotator names, comments, pilot items and annotation logs are not included.
- Known limits: 24 prompts; correlated samples; token limits differed between models; no guide rule for refusals.
