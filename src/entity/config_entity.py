from datetime import datetime
import os

from src.constant import training_pipeline
from src.logging.logger import logging


class TrainingPipelineConfig:
    def __init__(self):

        logging.info("Initializing Training Pipeline Config")

        self.timestamp = datetime.now().strftime(
            "%m_%d_%Y_%H_%M_%S"
        )

        self.pipeline_name = training_pipeline.PIPELINE_NAME

        self.artifact_dir = os.path.join(
            training_pipeline.ARTIFACT_DIR,
            self.timestamp
        )

        os.makedirs(self.artifact_dir, exist_ok=True)

        logging.info(f"Artifact directory created at: {self.artifact_dir}")

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        
        # Root directory for data ingestion artifacts
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )

        # Local dataset path
        self.dataset_path: str = training_pipeline.DATA_INGESTION_DATASET_PATH

        # Feature store path
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )

        # Train dataset path
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )

        # Test dataset path
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )

        # Train test split ratio
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

        ## Random State
        self.random_state: int = training_pipeline.RANDOM_STATE

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )

        self.validation_report_file_path = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_REPORT_FILE_NAME
        )

        self.validation_status_file_path = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_STATUS_FILE_NAME
        )

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        self.transformed_train_file_path = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_TRANSFORMED_FILE_NAME
        )

        self.transformed_test_file_path = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_TRANSFORMED_FILE_NAME
        )

        self.tokenizer_dir_path = os.path.join(
            self.data_transformation_dir,
            training_pipeline.TOKENIZER_DIR_NAME
        )