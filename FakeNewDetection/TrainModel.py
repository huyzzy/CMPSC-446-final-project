from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from data_preprocess import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import json
import os


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    cm = confusion_matrix(labels, preds).tolist()

    # ensure output folder exists
    os.makedirs("misinfo_model", exist_ok=True)

    results = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm
    }

    with open("misinfo_model/eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def train():
    df = load_dataset()

    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("label", ClassLabel(names=["real", "fake"]))
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tok(batch):
        return tokenizer(batch["content"], padding=True, truncation=True, max_length=256)

    dataset = dataset.map(tok, batched=True)
    dataset = dataset.remove_columns(["content"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    args = TrainingArguments(
        output_dir="misinfo_model",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_steps=500,
        logging_steps=500
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained("misinfo_model")
    tokenizer.save_pretrained("misinfo_model")


if __name__ == "__main__":
    train()
