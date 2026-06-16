from torch.utils.data import DataLoader

from src.configs.config import (
    TRAIN_CSV,
    TEST_CSV,
    IMAGE_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    MAX_QUESTION_LENGTH,
    NUM_WORKERS,
    TOP_K_ANSWERS,
)
from src.data.image_preprocessing import get_image_transform
from src.data.text_preprocessing import build_question_vocab, build_answer_vocab
from src.data.daquar_dataset import DAQUARDataset

def get_dataloader():
    word_to_idx, idx_to_word = build_question_vocab(
        csv_path=TRAIN_CSV,
        question_column="question",
        min_freq=1,
    )

    answer_to_idx, idx_to_answer = build_answer_vocab(
        csv_path=TRAIN_CSV,
        answer_column="answer",
        top_k=TOP_K_ANSWERS,
    )

    train_transform = get_image_transform(
        image_size=IMAGE_SIZE
    )
    test_transform = get_image_transform(
        image_size=IMAGE_SIZE
    )

    train_dataset = DAQUARDataset(
        csv_path=TRAIN_CSV,
        image_dir=IMAGE_DIR,
        word_to_idx=word_to_idx,
        answer_to_idx=answer_to_idx,
        transform=train_transform,
        question_column="question",
        answer_column="answer",
        image_column="image",
        max_length=MAX_QUESTION_LENGTH,
        filter_unknown_answer=True,
    )
    
    test_dataset = DAQUARDataset(
        csv_path=TEST_CSV,
        image_dir=IMAGE_DIR,
        word_to_idx=word_to_idx,
        answer_to_idx=answer_to_idx,
        transform=test_transform,
        question_column="question",
        answer_column="answer",
        image_column="image",
        max_length=MAX_QUESTION_LENGTH,
        filter_unknown_answer=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, test_loader, word_to_idx, idx_to_word, answer_to_idx, idx_to_answer
