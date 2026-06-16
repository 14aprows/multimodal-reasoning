from pathlib import Path 

ROOT_DIR = Path(__file__).parents[2]
DATA_DIR = ROOT_DIR / "dataset"
IMAGE_DIR = DATA_DIR / "images"

DATA_CSV = DATA_DIR / "data.csv"
TRAIN_CSV = DATA_DIR / "data_train.csv"
EVAL_CSV = DATA_DIR / "data_eval.csv"

TRAIN_IMAGES_LIST = DATA_DIR / "train_images_list.txt"
TEST_IMAGES_LIST = DATA_DIR / "test_images_list.txt"

ALL_QA = DATA_DIR / "all_qa_pairs.txt"
ANSWER_SPACE = DATA_DIR / "answer_space.txt"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_QUESTION_LENGTH = 20