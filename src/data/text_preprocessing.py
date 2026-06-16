import re
import pandas as pd
from collections import Counter
from configs.config import MAX_QUESTION_LENGTH, PAD_TOKEN, UNK_TOKEN

def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize_text(text):
    text = preprocess_text(text)
    tokens = text.split()
    return tokens

def build_question_vocab(csv_path, question_column="question", min_freq=1):
    df = pd.read_csv(csv_path)
    counter = Counter()
    for question in df[question_column]:
        tokens = tokenize_text(question)
        counter.update(tokens)

    word_to_idx = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for word, freq in counter.items():
        if freq >= min_freq:
            word_to_idx[word] = len(word_to_idx)

    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word

def build_answer_vocab(csv_path, answer_column="answer", top_k=None):
    df = pd.read_csv(csv_path)
    
    answers = []
    for answer in df[answer_column]:
        answer = preprocess_text(answer)
        answers.append(answer)
        
    counter = Counter(answers)
    if top_k is not None:
        counter = counter.most_common(top_k)
    else:
        counter = counter.most_common()

    answer_to_idx = {}

    for answer, _ in counter:
        answer_to_idx[answer] = len(answer_to_idx)

    idx_to_answer = {idx: answer for answer, idx in answer_to_idx.items()}
    return answer_to_idx, idx_to_answer

def encode_question(question, word_to_idx, max_length=MAX_QUESTION_LENGTH):
    tokens = tokenize_text(question)
    encoded = []
    for token in tokens:
        idx = word_to_idx.get(token, word_to_idx[UNK_TOKEN])
        encoded.append(idx)

    encoded = encoded[:max_length]
    while len(encoded) < max_length:
        encoded.append(word_to_idx[PAD_TOKEN])
    return encoded

def encode_answer(answer, answer_to_idx):
    answer = preprocess_text(answer)
    if answer not in answer_to_idx:
        return None
    return answer_to_idx[answer]
