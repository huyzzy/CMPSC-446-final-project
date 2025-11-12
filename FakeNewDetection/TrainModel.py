from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from data_preprocess import load_dataset

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
        eval_strategy="epoch",

        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["test"])
    trainer.train()

    model.save_pretrained("misinfo_model")
    tokenizer.save_pretrained("misinfo_model")

if __name__ == "__main__":
    train()
