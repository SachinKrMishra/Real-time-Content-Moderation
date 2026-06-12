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