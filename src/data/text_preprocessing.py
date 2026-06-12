import re
import pandas as pd
from collections import Counter

def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize_text(text):
    text = preprocess_text(text)
    tokens = text.split()
    return tokens

def build_vocab(tokens):
    