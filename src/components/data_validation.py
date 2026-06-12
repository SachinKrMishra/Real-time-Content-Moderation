import os, sys
import pandas as pd
from scipy.stats import ks_2samp

from src.logging.logger import logging
from src.exception.exception import ModerationException
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.constant import training_pipeline
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.utils.main_utils import(
    read_yaml_file,
    write_yaml_file,
)


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise ModerationException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ModerationException(e, sys)
    
    def validate_schema(self, dataframe: pd.DataFrame) -> bool:
        expected_columns = set(self.schema_config["columns"].keys())
        actual_columns = set(dataframe.columns)
        return expected_columns == actual_columns
    
    def validate_datatypes(self, dataframe: pd.DataFrame) -> bool:
        expected_columns = self.schema_config["columns"]

        for column, dtype in expected_columns.items():
            if str(dataframe[column].dtype) != dtype:
                logging.info(f"Datatype mismatch found in {column}")
                return False
        return True

    def validate_target_columns(self, dataframe: pd.DataFrame) -> bool:
        target_columns = self.schema_config["target_columns"]

        for col in target_columns:
            unique_values = set(dataframe[col].unique())
            if not unique_values.issubset({0, 1}):
                logging.info(f"Invalid values found in {col}")
                return False
        return True

    def validate_missing_values(self, dataframe: pd.DataFrame) -> bool:
        return bool(dataframe.isnull().sum().sum() == 0)
    
    def validate_empty_comments(self, dataframe: pd.DataFrame) -> bool:
        empty_comments =bool(
            dataframe[training_pipeline.TEXT_COLUMN]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        return empty_comments == 0

    def validate_dataset_size(self, dataframe: pd.DataFrame) -> bool:
        return len(dataframe) > 1000

    def get_duplicate_count(self, dataframe: pd.DataFrame) -> int:
        return int(dataframe.duplicated().sum() )

    def validate_text_length(self, dataframe: pd.DataFrame,max_length: int = 512) -> dict:
        lengths = (dataframe[training_pipeline.TEXT_COLUMN].astype(str).apply(len))

        return {
            "max_allowed_length":max_length,
            "samples_exceeding_limit":
                int(
                    (lengths > max_length)
                    .sum()
                )
        }

    def get_label_combinations(self, dataframe: pd.DataFrame):
        combinations = (
            dataframe[training_pipeline.TARGET_COLUMNS].astype(str)
            .agg(
                "_".join,
                axis=1
            )
            .value_counts()
            .head(20)
            .to_dict()
        )

        return combinations
    
    def get_text_statistics(self, dataframe: pd.DataFrame, text_column: str) -> dict:
        lengths = (
            dataframe[text_column]
            .astype(str)
            .apply(len)
        )

        return {
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "mean_length": round(float(lengths.mean()), 2),
            "median_length": round(float(lengths.median()), 2)
        }

    def get_label_distribution(self, dataframe: pd.DataFrame) -> dict:
        distribution = {}

        for col in training_pipeline.TARGET_COLUMNS:
            distribution[col] = round(float(dataframe[col].mean() * 100), 4)
        return distribution

    def get_class_imbalance_report(self, dataframe: pd.DataFrame) -> dict:
        report = {}
        total_rows = len(dataframe)

        for col in training_pipeline.TARGET_COLUMNS:
            positive_count = int(dataframe[col].sum())
            negative_count = (total_rows - positive_count)

            report[col] = {
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_ratio": round(positive_count /total_rows * 100,4)
            }

        return report
    
    @staticmethod
    def detect_dataset_drift(train_df: pd.DataFrame, test_df: pd.DataFrame, text_column: str = "comment_text"):
        """
        Detect drift between train and test datasets
        using Kolmogorov-Smirnov Test on text lengths.
        """

        train_lengths = train_df[text_column].astype(str).apply(len)
        test_lengths = test_df[text_column].astype(str).apply(len)

        statistic, p_value = ks_2samp(train_lengths, test_lengths)

        drift_detected = bool(p_value < 0.05)

        return {
            "ks_statistic": float(round(statistic, 6)),
            "p_value": float(round(p_value, 6)),
            "drift_detected": drift_detected
        }

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            drift_report = self.detect_dataset_drift(
            train_df=train_df, test_df=test_df, text_column=training_pipeline.TEXT_COLUMN
            )

            logging.info(f"Drift Detection Result: {drift_report}")

            validation_report = {}
            validation_status = True
            datasets = {
                "train": train_df,
                "test": test_df
            }

            for dataset_name, dataframe in datasets.items():
                schema_status = self.validate_schema(dataframe)

                datatype_status = self.validate_datatypes(dataframe)

                target_status = (self.validate_target_columns(dataframe))

                missing_status = (self.validate_missing_values(dataframe))

                empty_comment_status = (self.validate_empty_comments(dataframe))

                dataset_size_status = (self.validate_dataset_size(dataframe))

                duplicate_count = (self.get_duplicate_count(dataframe))

                text_statistics = (self.get_text_statistics(dataframe, training_pipeline.TEXT_COLUMN))

                text_length_validation = (self.validate_text_length(dataframe))

                label_distribution = (self.get_label_distribution(dataframe))

                class_imbalance_report = (self.get_class_imbalance_report(dataframe))

                label_combinations = (self.get_label_combinations(dataframe))

                validation_report[dataset_name] = {
                    "schema_validation": schema_status,
                    "datatype_validation": datatype_status,
                    "target_validation": target_status,
                    "missing_value_validation": missing_status,
                    "empty_comment_validation": empty_comment_status,
                    "dataset_size_validation": dataset_size_status,
                    "duplicate_count": duplicate_count,
                    "text_statistics": text_statistics,
                    "text_length_validation": text_length_validation,
                    "label_distribution": label_distribution,
                    "class_imbalance_report": class_imbalance_report,
                    "label_combinations": label_combinations
                }

                validation_status = (
                    validation_status
                    and schema_status
                    and datatype_status
                    and target_status
                    and missing_status
                    and empty_comment_status
                    and dataset_size_status
                )

            validation_report["dataset_drift"] = drift_report

            write_yaml_file(
                file_path=self.data_validation_config.validation_report_file_path,
                content=validation_report
            )

            logging.info("Data validation completed")

            return DataValidationArtifact(
                validation_status = validation_status,
                validation_report_file_path = self.data_validation_config.validation_report_file_path
            )

        except Exception as e:
            raise ModerationException(e, sys)