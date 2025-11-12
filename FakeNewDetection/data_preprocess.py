import pandas as pd

def load_dataset():
    # Load raw data
    buzz_fake = pd.read_csv("E:\Git Uploads\CMPSC-446-final-project\FakeNewDetection\example_article\BuzzFeed_fake_news_content.csv")
    buzz_real = pd.read_csv("E:\Git Uploads\CMPSC-446-final-project\FakeNewDetection\example_article\BuzzFeed_real_news_content.csv")
    poli_fake = pd.read_csv("E:\Git Uploads\CMPSC-446-final-project\FakeNewDetection\example_article\PolitiFact_fake_news_content.csv")
    poli_real = pd.read_csv("E:\Git Uploads\CMPSC-446-final-project\FakeNewDetection\example_article\PolitiFact_real_news_content.csv")

    # Label
    buzz_fake["label"] = 0
    poli_fake["label"] = 0
    buzz_real["label"] = 1
    poli_real["label"] = 1

    df = pd.concat([buzz_fake, buzz_real, poli_fake, poli_real], ignore_index=True)

    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    return df[["content", "label"]]
