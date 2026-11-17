"""Text preprocessing for French sentiment analysis (Allociné reviews).

Usage:
    from preprocess import clean_text, preprocess_text
"""
import re
import unicodedata

import spacy

__version__ = "1.0.0"

# Keep the parser and NER off: we only need tokenizer, tagger and lemmatizer.
nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])

# Negation words are useful for sentiment: never treat them as stop words.
NEGATIONS = {"ne", "n'", "pas", "plus", "jamais", "rien", "sans", "aucun"}
STOP = nlp.Defaults.stop_words - NEGATIONS


def clean_text(text: str) -> str:
    """Remove noise (URLs, HTML, mentions, repeated chars, extra spaces)."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"<[^>]+>", " ", text)                  # HTML tags
    text = re.sub(r"[@#]\w+", " ", text)                  # mentions / hashtags
    text = re.sub(r"([^\d\s])\1{2,}", r"\1", text)        # 3+ repeats -> 1
    text = re.sub(r"\s+", " ", text)                      # multiple spaces
    return text.strip()


def tokens_from_doc(doc, lemmas: bool = True, stop: bool = True) -> list[str]:
    """Turn an already-parsed spaCy Doc into a list of tokens."""
    out = []
    for tok in doc:
        if not (tok.is_alpha or tok.text == "n'"):
            continue
        w = tok.lemma_ if lemmas else tok.text
        if stop and (tok.text in STOP or w in STOP):
            continue
        out.append(w)
    return out


def normalize(text: str) -> str:
    """Clean, lowercase and use straight apostrophes (n’ -> n')."""
    return clean_text(text).lower().replace("’", "'")


def preprocess_text(text: str, lemmas: bool = True, stop: bool = True) -> list[str]:
    """Full pipeline for one text. For many texts use nlp.pipe + tokens_from_doc."""
    return tokens_from_doc(nlp(normalize(text)), lemmas, stop)
