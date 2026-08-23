from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer baseline")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--output", default="models/transformer")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    try:
        import evaluate
        import mlflow
        import numpy as np
        from datasets import load_dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit('Install optional dependencies: pip install -e ".[transformer]"') from exc

    root = Path(__file__).resolve().parents[1]
    files = {"train": str(root / args.train), "test": str(root / args.test)}
    dataset = load_dataset("csv", data_files=files)
    labels = sorted(set(dataset["train"]["label"]))
    label2id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=256)
        encoded["labels"] = [label2id[label] for label in batch["label"]]
        return encoded

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id
    )
    metric = evaluate.load("f1")

    def compute_metrics(prediction):
        predicted = np.argmax(prediction.predictions, axis=-1)
        return metric.compute(
            predictions=predicted, references=prediction.label_ids, average="macro"
        )

    output = root / args.output
    training_args = TrainingArguments(
        output_dir=str(output),
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    mlflow.set_tracking_uri((root / "mlruns").as_uri())
    mlflow.set_experiment("support-ticket-intent-transformer")
    with mlflow.start_run(run_name=args.model):
        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_params(
            {"base_model": args.model, "epochs": args.epochs, "batch_size": args.batch_size}
        )
        mlflow.log_metrics(
            {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
        )
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    (output / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
