# Model leaderboard

Generated 2026-08-11 16:58 UTC by `python -m training.compare_runs --md`, ranked by `macro_f1`, one row per model (its latest completed run). Source: MLflow (`mlflow.db`, local/git-ignored) -- re-run after training to refresh.

**Best model: `rnn_lstm`** (`macro_f1` = 0.6488).

```
Run ID   : 6927b8d15cd14d3a89adc092ae76eb17
Model    : rnn_lstm
Accuracy : 0.7060
ROC-AUC  : 0.8614
F1       : 0.6488

Run ID   : 0b082c33b1c8412ab0f84d1a52124adc
Model    : google/bert_uncased_L-4_H-256_A-4
Accuracy : 0.7093
ROC-AUC  : 0.8609
F1       : 0.6461

Run ID   : 40c573c1996b422fbc97872e7b405377
Model    : logreg
Accuracy : 0.6896
ROC-AUC  : 0.8493
F1       : 0.6252

Run ID   : 045cb48a33534dc787ea3ebe497bf6b0
Model    : linear_svc
Accuracy : 0.7206
ROC-AUC  : -
F1       : 0.6225
```
