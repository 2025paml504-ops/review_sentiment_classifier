# Model leaderboard

Generated 2026-08-12 01:15 UTC by `python -m training.compare_runs --md`, ranked by `macro_f1`, one row per model (its best completed run). Source: MLflow (`mlflow.db`, local/git-ignored) -- re-run after training to refresh.

**Best model: `google/bert_uncased_L-4_H-256_A-4`** (`macro_f1` = 0.6685).

```
Run ID   : b94dd819d40f416a89f8b2ee3b5d83ce
Model    : google/bert_uncased_L-4_H-256_A-4
Accuracy : 0.6832
ROC-AUC  : 0.8462
F1       : 0.6685

Run ID   : 5ec93d9934b44daa922b4e0e193776e8
Model    : rnn_lstm
Accuracy : 0.6625
ROC-AUC  : 0.8394
F1       : 0.6568

Run ID   : 71e7c534f96f4b10ae37aa05797a9706
Model    : logreg
Accuracy : 0.6568
ROC-AUC  : 0.8340
F1       : 0.6470

Run ID   : 96716446f6d548408144b9c4ec7c0c48
Model    : linear_svc
Accuracy : 0.6508
ROC-AUC  : -
F1       : 0.6341
```
