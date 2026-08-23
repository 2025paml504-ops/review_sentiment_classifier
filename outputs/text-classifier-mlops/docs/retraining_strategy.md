# Monitoring and retraining strategy

## Signals

- **Input drift:** token OOV rate and population stability index (PSI) for token count.
- **Output drift:** Jensen-Shannon divergence of predicted class proportions.
- **Uncertainty:** share of predictions below the confidence threshold.
- **Performance:** macro-F1 once delayed labels arrive; this is the decisive quality signal.

Drift metrics are warning signals, not ground-truth performance. Seasonality, campaigns, outages, and routing changes can cause legitimate distribution shifts.

## Trigger

Wait for the configured minimum event count. Open a retraining candidate if two unsupervised signals breach their thresholds, or immediately when labeled macro-F1 is below `minimum_macro_f1`. Alert a human owner and record the window, model version, thresholds, and data lineage.

## Safe retraining workflow

1. Snapshot and version recent examples and delayed labels; redact PII and validate schema.
2. Diagnose slices by class, channel, language, region, and confidence. Correct labeling problems first.
3. Add a representative sample to training while retaining a frozen regression set and a recent temporal holdout.
4. Re-run classical and transformer experiments with pinned configuration and seed.
5. Require the candidate to beat the champion on macro-F1, avoid material per-class regression, pass latency/size limits, and pass API/security tests.
6. Deploy as a canary, compare live metrics, then promote or roll back. Keep the previous model artifact available.

## Cadence and ownership

Run drift checks daily or per traffic window and evaluate delayed labels weekly. Review thresholds monthly after enough baseline history exists. The ML owner approves model quality, support operations validates label semantics, and the service owner controls deployment and rollback.

