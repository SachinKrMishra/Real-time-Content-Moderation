import os, sys
import pickle
import dill
import yaml
import numpy as np
import pandas as pd
from src.constant import training_pipeline
from src.exception.exception import ModerationException

def validate_multilabel_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    '''
        Compare label distributions between train and test datasets.
    '''
    result = []
    for col in training_pipeline.TARGET_COLUMNS:
        train_percentage = train_df[col].mean()*100
        test_percentage = test_df[col].mean()*100

        difference = abs(train_percentage - test_percentage)

        result.append({
            "label": col,
            "train_percentage": round(train_percentage, 4),
            "test_percentage": round(test_percentage, 4),
            "difference": round(difference, 4)
        })

    return pd.DataFrame(result)

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ModerationException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        
        with open(file_path, 'w') as file:
            yaml.dump(content, file)
    
    except Exception as e:
        raise ModerationException(e, sys)
    
def save_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise ModerationException(e, sys)


def load_object(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise ModerationException(e, sys)