from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from data_preprocess import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import json
import os


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    cm = confusion_matrix(labels, preds).tolist()

    # we'll also write this to disk later
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def train():
    # 1. Load combined dataset (all 6 CSVs) from data_preprocess.load_dataset()
    df = load_dataset()  # returns columns: content (text), label (0/1)

    # 2. Wrap in HF Dataset and split
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tok(batch):
        return tokenizer(
            batch["content"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    dataset = dataset.map(tok, batched=True)

    # Remove any non-tensor columns (like "content", index, etc.)
    keep_cols = ["input_ids", "attention_mask", "label"]
    remove_cols = [c for c in dataset["train"].column_names if c not in keep_cols]
    dataset = dataset.remove_columns(remove_cols)

    # Trainer expects "labels" column
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")

    # 4. Model: 2 labels (0 = fake, 1 = real)
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2,
    )

    # 5. Training arguments – NO evaluation_strategy here
    args = TrainingArguments(
        output_dir="misinfo_model_ckpts",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        do_train=True,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    # 6. Train
    trainer.train()

    # 7. Evaluate once at the end
    metrics = trainer.evaluate()

    # 8. Save final model where inference.py expects it
    os.makedirs("misinfo_model", exist_ok=True)
    model.save_pretrained("misinfo_model")
    tokenizer.save_pretrained("misinfo_model")

    # Save metrics
    with open(os.path.join("misinfo_model", "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    train()
