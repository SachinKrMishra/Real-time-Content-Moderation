from src.logging.logger import logging
from src.exception.exception import ModerationException

## Configuration of the data ingestion config
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.constant import training_pipeline


import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from src.utils.main_utils import validate_multilabel_split

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ModerationException(e, sys)
        
    def export_data_as_dataframe(self):
        '''
        Read data from local CSV file
        '''
        try:
            file_path = self.data_ingestion_config.dataset_path

            logging.info(f"Raeding dataset from path : {file_path}")

            df = pd.read_csv(file_path)

            df = df.drop(columns=['id'], errors='ignore')

            logging.info(f"Dataset loaded successfully with shape {df.shape}")

            return df
        
        except Exception as e:
            raise ModerationException(e, sys)
        
    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            logging.info(f"Data exported to feature store at {feature_store_file_path}")

            return dataframe
        
        except Exception as e:
            raise ModerationException(e, sys)
        
    def split_data_as_train_test_split(self,dataframe: pd.DataFrame):
        try:
            X = dataframe.drop(columns=list(training_pipeline.TARGET_COLUMNS))

            y = dataframe[list(training_pipeline.TARGET_COLUMNS)]

            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=self.data_ingestion_config.random_state
            )

            train_idx, test_idx = next(msss.split(X, y))

            train_set = dataframe.iloc[train_idx]
            test_set = dataframe.iloc[test_idx]

            split_validation_df = validate_multilabel_split(train_set, test_set)
            logging.info(f"\n{split_validation_df.to_string(index=False)}")

            max_difference = split_validation_df["difference"].max()

            if max_difference > 0.5:
                raise Exception(
                    f"Split validation failed. "
                    f"Maximum label difference = {max_difference}%"
                )    

            logging.info(f"Train shape: {train_set.shape}")

            logging.info(f"Test shape: {test_set.shape}")

            train_dir = os.path.dirname(self.data_ingestion_config.training_file_path)

            test_dir = os.path.dirname(self.data_ingestion_config.testing_file_path)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False
            )

            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False
            )

            logging.info("Multi-label stratified train-test split completed")

        except Exception as e:
            raise ModerationException(e, sys)

    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_data_as_dataframe()

            dataframe = self.export_data_into_feature_store(dataframe)

            self.split_data_as_train_test_split(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info("Data ingestion completed")

            return data_ingestion_artifact

        except Exception as e:
            raise ModerationException(e, sys)
        
        