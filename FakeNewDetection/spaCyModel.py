import spacy
from spacy.tokens import Doc, Span
from spacy.language import Language
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#RUN THIS COMMAND IF NEEDED
#python -m spacy download en_core_web_lg

vader = SentimentIntensityAnalyzer()

EXAGGERATION = {"unbelievable","massive","shocking","incredible","historic"}
FEAR = {"crisis","collapse","emergency","panic","deadly","catastrophic"}
BIAS = {"obviously","clearly","must","should","have","need"}

# Register extension if not already registered
if not Doc.has_extension("persuasive_spans"):
    Doc.set_extension("persuasive_spans", default=[])
if not Doc.has_extension("sentiment"):
    Doc.set_extension("sentiment", default=None)

@Language.component("persuasion_component")
def persuasion_component(doc):
    spans = []
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in EXAGGERATION:
            spans.append(("EXAGGERATION", token.i, token.i+1))
        elif lemma in FEAR:
            spans.append(("FEAR_APPEAL", token.i, token.i+1))
        elif lemma in BIAS:
            spans.append(("BIAS_MODALITY", token.i, token.i+1))

    doc._.persuasive_spans = [Span(doc, s, e, label=l) for l, s, e in spans]
    sentiment_scores = vader.polarity_scores(doc.text)
    doc._.sentiment = sentiment_scores
    return doc

def load_spacy():
    nlp = spacy.load("en_core_web_lg")
    # Add the component by NAME (spaCy 3.x requirement)
    nlp.add_pipe("persuasion_component", last=True)
    return nlp
