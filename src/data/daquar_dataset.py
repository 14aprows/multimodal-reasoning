from pathlib import Path
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.configs.config import MAX_QUESTION_LENGTH
from src.data.text_preprocessing import encode_question, encode_answer

class DAQUARDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_dir,
        word_to_idx,
        answer_to_idx,
        transform=None,
        question_column="question",
        answer_column="answer",
        image_column="image",
        max_length=MAX_QUESTION_LENGTH,
        filter_unknown_answer=False
    ):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.word_to_idx = word_to_idx
        self.answer_to_idx = answer_to_idx
        self.transform = transform
        self.question_column = question_column
        self.answer_column = answer_column
        self.image_column = image_column
        self.max_length = max_length

        self.df = pd.read_csv(csv_path)
        if filter_unknown_answer:
            self.df = self.df[
                self.df[self.answer_column].apply(
                    lambda x: encode_answer(x, self.answer_to_idx) is not None
                )
            ].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name = str(row[self.image_column])
        question = str(row[self.question_column])
        answer = str(row[self.answer_column])

        image_path = self.image_dir / image_name
        if not image_path.exists():
            possible_paths = [
                self.image_dir / f"{image_name}.jpg",
                self.image_dir / f"{image_name}.png",
                self.image_dir / f"{image_name}.jpeg",
            ]

            image_path = None
            for path in possible_paths:
                if path.exists():
                    image_path = path
                    break
            if image_path is None:
                raise FileNotFoundError(f"Image not found: {image_name}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        question_encoded = encode_question(
            question,
            self.word_to_idx,
            self.max_length
        )
        answer_encoded = encode_answer(
            answer,
            self.answer_to_idx
        )
        if answer_encoded is None:
            raise ValueError(f"Unknown answer: {answer}")

        question_tensor = torch.tensor(question_encoded, dtype=torch.long)
        answer_tensor = torch.tensor(answer_encoded, dtype=torch.long)
        return {
            "image": image,
            "question": question_tensor,
            "answer": answer_tensor,
        }
