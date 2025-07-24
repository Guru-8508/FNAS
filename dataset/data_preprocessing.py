import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_data(self):
        """Load the thermal face dataset"""
        try:
            # Use absolute path to avoid file not found errors
            if not os.path.isabs(self.data_path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                self.data_path = os.path.join(base_dir, self.data_path)
            
            print(f"Loading data from: {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully! Shape: {self.df.shape}")
            return self.df
            
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}")
            print("Please ensure the CSV file exists in the correct location.")
            return None
    
    def explore_data(self):
        """Basic data exploration"""
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumn names: {list(self.df.columns)}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.df.describe()}")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates}")
        
        return self.df.head()
    
    def clean_data(self):
        """Clean the dataset"""
        print("\n=== DATA CLEANING ===")
        original_shape = self.df.shape
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        print(f"Removed {original_shape[0] - self.df.shape[0]} duplicate rows")
        
        # Handle missing values
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with mean
        if len(numeric_columns) > 0:
            self.df[numeric_columns] = self.imputer.fit_transform(self.df[numeric_columns])
            print(f"Filled missing values in numeric columns: {list(numeric_columns)}")
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if self.df[col].isnull().sum() > 0:
                mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                self.df[col].fillna(mode_value, inplace=True)
                print(f"Filled missing values in {col} with: {mode_value}")
        
        print(f"Final dataset shape: {self.df.shape}")
        return self.df
    
    def feature_engineering(self, target_column=None):
        """Perform feature engineering"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Identify target column if not specified
        if target_column is None:
            # Common target column names for thermal face detection
            possible_targets = ['label', 'class', 'target', 'category', 'is_face', 'face_detected']
            for col in possible_targets:
                if col in self.df.columns:
                    target_column = col
                    break
        
        if target_column and target_column in self.df.columns:
            print(f"Using '{target_column}' as target variable")
            
            # Encode target variable if it's categorical
            if self.df[target_column].dtype == 'object':
                self.df[target_column] = self.label_encoder.fit_transform(self.df[target_column])
                print(f"Encoded target variable. Classes: {self.label_encoder.classes_}")
        
        # Handle categorical features
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != target_column]
        
        for col in categorical_columns:
            if self.df[col].nunique() < 10:  # One-hot encode if few categories
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)
                print(f"One-hot encoded: {col}")
            else:  # Label encode if many categories
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                print(f"Label encoded: {col}")
        
        return self.df, target_column
    
    def scale_features(self, target_column=None):
        """Scale numerical features"""
        print("\n=== FEATURE SCALING ===")
        
        # Separate features and target
        if target_column and target_column in self.df.columns:
            X = self.df.drop(target_column, axis=1)
            y = self.df[target_column]
        else:
            X = self.df
            y = None
        
        # Scale numerical features
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            X[numeric_columns] = self.scaler.fit_transform(X[numeric_columns])
            print(f"Scaled features: {list(numeric_columns)}")
        
        return X, y
    
    def split_data(self, X, y=None, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\n=== DATA SPLITTING ===")
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            print(f"Training set: {X_train.shape}")
            print(f"Test set: {X_test.shape}")
            print(f"Target distribution in training set:\n{pd.Series(y_train).value_counts()}")
            
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
            print(f"Training set: {X_train.shape}")
            print(f"Test set: {X_test.shape}")
            
            return X_train, X_test, None, None
    
    def save_processed_data(self, X_train, X_test, y_train=None, y_test=None, output_dir="processed_data"):
        """Save processed data to files"""
        print(f"\n=== SAVING PROCESSED DATA ===")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training and test sets
        X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
        
        if y_train is not None:
            pd.Series(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
            pd.Series(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
        
        print(f"Processed data saved to: {output_dir}/")
        
        return {
            'X_train_path': os.path.join(output_dir, "X_train.csv"),
            'X_test_path': os.path.join(output_dir, "X_test.csv"),
            'y_train_path': os.path.join(output_dir, "y_train.csv") if y_train is not None else None,
            'y_test_path': os.path.join(output_dir, "y_test.csv") if y_test is not None else None
        }

# Main preprocessing function
def preprocess_thermal_dataset(data_path="thermal_face_dataset.csv", target_column=None):
    """Complete preprocessing pipeline"""
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_path)
    
    # Load data
    df = preprocessor.load_data()
    if df is None:
        return None
    
    # Explore data
    preprocessor.explore_data()
    
    # Clean data
    preprocessor.clean_data()
    
    # Feature engineering
    df, target_col = preprocessor.feature_engineering(target_column)
    
    # Scale features
    X, y = preprocessor.scale_features(target_col)
    
    # Split data
    if y is not None:
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    else:
        X_train, X_test, y_train, y_test = preprocessor.split_data(X)
    
    # Save processed data
    file_paths = preprocessor.save_processed_data(X_train, X_test, y_train, y_test)
    
    print("\n=== PREPROCESSING COMPLETE ===")
    print("Your data is now ready for model training!")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'file_paths': file_paths,
        'preprocessor': preprocessor
    }

# Usage example
if __name__ == "__main__":
    # Process the thermal face dataset
    result = preprocess_thermal_dataset(
    data_path=r"D:/FNAS/dataset/thermal_face_dataset.csv",  # Use full absolute path
    target_column="label"
)

    
    if result:
        print("\nYour processed data is ready for machine learning models!")
        print(f"Training features shape: {result['X_train'].shape}")
        if result['y_train'] is not None:
            print(f"Training labels shape: {result['y_train'].shape}")
