import os
import sys
import re
import emoji
import contractions
import pandas as pd
from transformers import AutoTokenizer

from src.logging.logger import logging
from src.exception.exception import ModerationException
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact
)

from src.entity.config_entity import DataTransformationConfig
from src.constant import training_pipeline

class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig
    ):

        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    def read_data(self, file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ModerationException(e, sys)

    def clean_text(self, text):
        text = str(text)
        text = text.lower()

        # URL removal
        text = re.sub( r"https?://\S+|www\.\S+",  "", text)

        # HTML removal
        text = re.sub(r"<.*?>", "", text)

        # contractions
        text = contractions.fix(text)

        # emoji normalization
        text = emoji.demojize(text, delimiters=(" ", " "))

        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(training_pipeline.MODEL_NAME)
        tokenizer.save_pretrained(self.data_transformation_config.tokenizer_dir_path)
        return tokenizer

    def clean_dataframe(self, dataframe):
        dataframe[training_pipeline.TEXT_COLUMN] = (
            dataframe[training_pipeline.TEXT_COLUMN]
            .fillna("")
            .apply(self.clean_text)
        )

        return dataframe

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation")
            if not self.data_validation_artifact.validation_status:
                raise Exception("Validation failed")

            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            
            train_df = self.clean_dataframe(train_df)
            test_df = self.clean_dataframe(test_df)

            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.transformed_train_file_path
                ),
                exist_ok=True
            )

            train_df.to_csv(self.data_transformation_config.transformed_train_file_path, index=False)
            test_df.to_csv(self.data_transformation_config.transformed_test_file_path, index=False)

            self.get_tokenizer()

            logging.info("Data Transformation completed")

            return DataTransformationArtifact(
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path,
                tokenizer_dir_path = self.data_transformation_config.tokenizer_dir_path
            )

        except Exception as e:
            raise ModerationException(e,sys)