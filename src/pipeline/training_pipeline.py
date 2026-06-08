import os
import sys

from  src.logging.logger import logging
from src.exception.exception import ModerationException

from src.components.data_ingestion import DataIngestion

from src.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
)

from src.entity.artifact_entity import DataIngestionArtifact

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