import os

'''
Defining common constant variable for training pipeline.
'''

PIPELINE_NAME = "ContentModeration"
ARTIFACT_DIR = "artifacts"

TEXT_COLUMN = "comment_text"

TARGET_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
)

SCHEMA_FILE_PATH = os.path.join(
    "data_schema",
    "schema.yaml"
)

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TEST_LABEL_FILE_NAME = "test_labels.csv"

NUM_LABELS = len(TARGET_COLUMNS)

'''
Data Ingestion related constants.
'''

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_DATASET_PATH: str = "raw_data/train.csv"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR:str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
RANDOM_STATE: int = 42