# Model leaderboard

Generated 2026-08-20 11:20 UTC by `python -m training.compare_runs --md`, ranked by `macro_f1`, one row per model (its best completed run). Source: MLflow (`mlflow.db`, local/git-ignored) -- re-run after training to refresh.

**Best model: `rnn_lstm`** (`macro_f1` = 0.8918).

```
Run ID   : 964de3888934451e85f23102f141d7d5
Started  : 2026-08-20 09:49
Model    : rnn_lstm
Accuracy : 0.9434
ROC-AUC  : 0.9849
F1       : 0.8918

Run ID   : 7865468f90274fd5a7d639ed96832836
Started  : 2026-08-20 09:47
Model    : linear_svc
Accuracy : 0.9412
ROC-AUC  : 0.9671
F1       : 0.8652

Run ID   : 97bb3982e0f64ed8963063fdc22e7219
Started  : 2026-08-20 10:25
Model    : google/bert_uncased_L-2_H-128_A-2
Accuracy : 0.9208
ROC-AUC  : 0.9704
F1       : 0.8519

Run ID   : 959e7569dd5441a4a6f4e2a7b1f03276
Started  : 2026-08-20 11:16
Model    : logreg
Accuracy : 0.9073
ROC-AUC  : 0.9690
F1       : 0.8358
```
