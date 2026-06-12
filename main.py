from src.pipeline.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()

data_ingestion_artifact = (pipeline.start_data_ingestion())
data_validation_artifact = (pipeline.start_data_validation(data_ingestion_artifact))