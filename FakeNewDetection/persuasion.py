# FakeNewDetection/persuasion.py

import re
from typing import List, Tuple

import spacy
from spacy.tokens import Doc

# Load a larger spaCy model for better vectors / POS / NER
# Make sure you've run:  python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")

# Simple lexicons for different persuasive strategies
INTENSIFIERS = {
    "very", "extremely", "incredibly", "unbelievably", "absolutely",
    "totally", "completely", "highly", "deeply", "hugely", "truly"
}

FEAR_WORDS = {
    "threat", "threaten", "danger", "dangerous", "crisis", "disaster",
    "catastrophe", "collapse", "ruin", "invasion", "extinction", "panic"
}

MORALIZING = {
    "evil", "corrupt", "immoral", "disgusting", "unacceptable",
    "outrageous", "shameful", "disgraceful"
}

AUTHORITY_PHRASES = {
    "experts say", "scientists agree", "research shows",
    "studies prove", "everyone knows", "people are saying"
}


def _find_caps_cues(doc: Doc) -> List[Tuple[str, str]]:
    """Words in ALL CAPS (except tiny ones like USA) → often emphasis / shouting."""
    cues = []
    for token in doc:
        if token.is_alpha and token.text.isupper() and len(token.text) > 3:
            cues.append((token.text, "ALL_CAPS_EMPHASIS"))
    return cues


def _find_punctuation_cues(text: str) -> List[Tuple[str, str]]:
    """Multiple exclamation/question marks."""
    cues = []
    if re.search(r"!!!+", text):
        cues.append(("!!!", "EXCESSIVE_EXCLAMATION"))
    if re.search(r"\?\?+", text):
        cues.append(("??", "EXCESSIVE_QUESTIONING"))
    return cues


def _find_lexical_cues(doc: Doc) -> List[Tuple[str, str]]:
    """Use spaCy tokens + lemmas to find fear words, intensifiers, moral language."""
    cues = []
    for token in doc:
        lemma = token.lemma_.lower()
        text = token.text

        if lemma in INTENSIFIERS:
            # intensifier + following adjective/adverb if present
            span_text = text
            if token.i + 1 < len(doc) and doc[token.i + 1].pos_ in {"ADJ", "ADV"}:
                span_text += " " + doc[token.i + 1].text
            cues.append((span_text, "INTENSIFIER"))

        if lemma in FEAR_WORDS:
            cues.append((text, "FEAR_APPEAL"))

        if lemma in MORALIZING:
            cues.append((text, "MORAL_JUDGMENT"))

    return cues


def _find_authority_cues(doc: Doc) -> List[Tuple[str, str]]:
    """Look for authority phrases at sentence level."""
    cues = []
    lower_text = doc.text.lower()
    for phrase in AUTHORITY_PHRASES:
        if phrase in lower_text:
            cues.append((phrase, "APPEAL_TO_AUTHORITY"))
    return cues


def _dedupe(cues: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    result = []
    for text, tag in cues:
        key = (text.strip(), tag)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def extract_persuasive_cues(text: str) -> List[Tuple[str, str]]:
    """
    Main entry point used by inference.py.
    Returns list of (cue_text, cue_tag).
    """
    doc = nlp(text)

    cues = []
    cues.extend(_find_caps_cues(doc))
    cues.extend(_find_punctuation_cues(text))
    cues.extend(_find_lexical_cues(doc))
    cues.extend(_find_authority_cues(doc))

    return sorted(_dedupe(cues), key=lambda x: x[1])

