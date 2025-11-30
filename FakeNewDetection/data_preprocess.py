import pandas as pd
from pathlib import Path


def load_dataset():
    root = Path(__file__).resolve().parent
    data_dir = root / "example_article"

    # --- Load all six CSVs ---
    buzz_fake = pd.read_csv(data_dir / "BuzzFeed_fake_news_content.csv")
    buzz_real = pd.read_csv(data_dir / "BuzzFeed_real_news_content.csv")
    poli_fake = pd.read_csv(data_dir / "PolitiFact_fake_news_content.csv")
    poli_real = pd.read_csv(data_dir / "PolitiFact_real_news_content.csv")
    fake_news = pd.read_csv(data_dir / "FakeNews.csv")
    real_news = pd.read_csv(data_dir / "RealNews.csv")

    # --- Assign labels: 0 = FAKE, 1 = REAL ---
    for df in (buzz_fake, poli_fake, fake_news):
        df["label"] = 0

    for df in (buzz_real, poli_real, real_news):
        df["label"] = 1

    df = pd.concat(
        [buzz_fake, buzz_real, poli_fake, poli_real, fake_news, real_news],
        ignore_index=True,
    )
    df["content"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["text"].fillna("").astype(str)
    )

    # Only keep what we need
    return df[["content", "label"]]

