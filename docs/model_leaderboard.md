# Model leaderboard

Generated 2026-08-13 10:00 UTC by `python -m training.compare_runs --md`, ranked by `macro_f1`, one row per model (its best completed run). Source: MLflow (`mlflow.db`, local/git-ignored) -- re-run after training to refresh.

**Best model: `rnn_lstm`** (`macro_f1` = 0.6488).

```
Run ID   : 9f0f73731f404cd6a3267aa03cb48943
Started  : 2026-08-13 05:53
Model    : rnn_lstm
Accuracy : 0.7060
ROC-AUC  : 0.8614
F1       : 0.6488

Run ID   : c366fa9a434547a3bde151823eb940fd
Started  : 2026-08-13 07:19
Model    : google/bert_uncased_L-4_H-256_A-4
Accuracy : 0.7093
ROC-AUC  : 0.8609
F1       : 0.6461

Run ID   : 2ab4fad61a4c4137be7f32ea1eef9144
Started  : 2026-08-13 05:31
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
