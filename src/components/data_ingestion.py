import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split  # Fixed typo: was train_text_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load your thermal face dataset
            df = pd.read_csv(r"D:\FNAS\dataset\thermal_face_dataset.csv")  # Your generated thermal dataset
            logging.info("Read the thermal dataset as dataframe")
            
            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Saved raw thermal data to artifacts folder")
            
            # Split the data (80% train, 20% test)
            logging.info("Initiating train test split")
            train_set, test_set = train_test_split(
                df, 
                test_size=0.2, 
                random_state=42,
                stratify=df['deficiency_label']  # Ensure balanced split
            )
            
            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data ingestion completed successfully")
            logging.info(f"Train set shape: {train_set.shape}")
            logging.info(f"Test set shape: {test_set.shape}")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(f"Train data saved at: {train_data_path}")
    print(f"Test data saved at: {test_data_path}")
