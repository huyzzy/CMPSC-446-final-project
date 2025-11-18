import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from spaCyModel import load_spacy

tokenizer = AutoTokenizer.from_pretrained("misinfo_model")
model = AutoModelForSequenceClassification.from_pretrained("misinfo_model")
nlp = load_spacy()

def confidence_bar(conf):
    """
    Creates a simple progress bar for confidence 0.0–1.0
    """
    length = 20  # bar width
    filled = int(conf * length)
    bar = "█" * filled + "░" * (length - filled)
    percent = int(conf * 100)
    return f"{bar} {percent}%"

def highlight_cues(text, cues):
    """
    Highlights persuasive cue words inside the article text.
    """
    highlighted = text
    for word, tag in cues:
        highlighted = highlighted.replace(
            word,
            f"[{word.upper()}]"
        )
    return highlighted

def analyze(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    logits = model(**enc).logits
    probs = F.softmax(logits, dim=1)

    label = "REAL" if torch.argmax(probs) == 1 else "FAKE"
    conf = float(probs[0, torch.argmax(probs)].detach())  # detach fixes warning

    doc = nlp(text)
    cues = [(span.text, span.label_) for span in doc._.persuasive_spans]

    return label, conf, cues
