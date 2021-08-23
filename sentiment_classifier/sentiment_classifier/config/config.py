
SELECTED_FEATURES = ["review_comment_title",'review_comment_message']

INPUT_FEATURES = ["review_comment_title",'review_comment_message','review_score']

DEVICE = 'cpu'

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

VOCAB_PATH = ROOT_DIR / 'trained_models' / 'wordtoidx.pkl'

WEIGHTS_PATH = ROOT_DIR / 'trained_models' / 'model_weights'

DATASETS_DIR = ROOT_DIR/ 'datasets'

