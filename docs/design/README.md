# Design docs

[← Back to project README](../../README.md)

These documents explain **how the system is structured and why** — the design
decisions behind the pipeline, not just how to run it. For run/contribute
instructions see the [top-level docs](../pipeline.md).

## Contents

- **[Architecture guide](architecture.md)** — components, data flow (the DVC DAG),
  the serving API and UI, tech stack, design principles, and where future work
  plugs in.
- **[Decision-making guide](decisions.md)** — the key choices (text cleaning,
  sentiment labeling, TF-IDF, SQLite feature store, DVC, which model gets
  served, the serving API, the UI, …) with the context, rationale, and
  alternatives considered for each.
