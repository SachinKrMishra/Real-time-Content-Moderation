from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str 
    train_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    tokenizer_dir_path: str