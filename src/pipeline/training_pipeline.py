import os
import sys

from  src.logging.logger import logging
from src.exception.exception import ModerationException

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation

from src.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig
)

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            logging.info("Initiate Data Ingestion")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f'Data Initiation completed and artifacts : {data_ingestion_artifact}')
            return data_ingestion_artifact

        except Exception as e:
            raise ModerationException(e,sys)
    
    def start_data_validation(self, data_ingestion_artifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            logging.info("Initiate Data Validation")
            data_validation_artifact = (data_validation.initiate_data_validation())
            logging.info(f"Data Validation completed and artifacts : "f"{data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise ModerationException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact):
        try:

            if not data_validation_artifact.validation_status:
                raise Exception("Data validation failed. Cannot start transformation.")

            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )

            logging.info("Initiate Data Transformation")

            data_transformation_artifact = (data_transformation.initiate_data_transformation())

            logging.info(
                f"Data Transformation completed and artifacts: "
                f"{data_transformation_artifact}"
            )

            return data_transformation_artifact

        except Exception as e:
            raise ModerationException(e, sys)