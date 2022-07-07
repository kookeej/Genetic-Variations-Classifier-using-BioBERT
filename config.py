import transformers
from transformers import AutoConfig

import re

class DefaultConfig:
    # hyperparameters
    RANDOM_SEED = 42
    TRAIN_BATCH = 32
    VALID_BATCH = 32
    TEST_BATCH = 64
    SEED = 42

    EPOCHS = 10
    FOLDS = 5

    # model options
    MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1" # "microsoft/deberta-base" 
    MODEL_CONFIG = AutoConfig.from_pretrained(MODEL_NAME)
    OPTION = ""

    ID_DICT_LEN = 3316
    TRAIN_LOG_INTERVAL = 1
    VALID_LOG_INTERVAL = 1