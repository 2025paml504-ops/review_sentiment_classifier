# Model leaderboard

Generated 2026-08-12 17:55 UTC by `python -m training.compare_runs --md`, ranked by `macro_f1`, one row per model (its best completed run). Source: MLflow (`mlflow.db`, local/git-ignored) -- re-run after training to refresh.

**Best model: `rnn_lstm`** (`macro_f1` = 0.6488).

```
Run ID   : 0033ab99382449caa09105e50d9efa23
Started  : 2026-08-11 17:27
Model    : rnn_lstm
Accuracy : 0.7060
ROC-AUC  : 0.8614
F1       : 0.6488

Run ID   : 90d3208fecd4455796be3b5b829e1ea2
Started  : 2026-08-12 01:49
Model    : google/bert_uncased_L-4_H-256_A-4
Accuracy : 0.7093
ROC-AUC  : 0.8609
F1       : 0.6461

Run ID   : a557ecf261e9443ab1e72bb93edb7df0
Started  : 2026-08-11 17:11
Model    : logreg
Accuracy : 0.6896
ROC-AUC  : 0.8493
F1       : 0.6252

Run ID   : 0f89cabfb6d94fbb995e3de93827bc50
Started  : 2026-08-11 17:25
Model    : linear_svc
Accuracy : 0.7206
ROC-AUC  : -
F1       : 0.6225
```
