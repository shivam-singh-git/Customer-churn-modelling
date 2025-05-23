import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None  # Store feature columns used during training
        
    def load_data(self, train_path, test_path):
        """Load training and testing datasets"""
        try:
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Training data file not found at {train_path}")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Testing data file not found at {test_path}")
                
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            if train_data.empty or test_data.empty:
                raise ValueError("Empty dataset loaded")
                
            return train_data, test_data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            # Fill numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
            
            return df
        except Exception as e:
            raise Exception(f"Error handling missing values: {str(e)}")
    
    def encode_categorical_features(self, df):
        """Encode categorical features using Label Encoding"""
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            
            return df
        except Exception as e:
            raise Exception(f"Error encoding categorical features: {str(e)}")
    
    def normalize_numerical_features(self, df):
        """Normalize numerical features using StandardScaler"""
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            return df
        except Exception as e:
            raise Exception(f"Error normalizing numerical features: {str(e)}")
    
    def preprocess_data(self, train_data, test_data, target_column):
        """Complete preprocessing pipeline"""
        try:
            if not isinstance(train_data, pd.DataFrame) or not isinstance(test_data, pd.DataFrame):
                raise ValueError("Input must be pandas DataFrames")
                
            if target_column not in train_data.columns or target_column not in test_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Store target variables and convert to integer
            y_train = train_data[target_column].copy()
            y_test = test_data[target_column].copy()
            
            # Handle missing values in target
            y_train = y_train.fillna(y_train.mode().iloc[0])
            y_test = y_test.fillna(y_test.mode().iloc[0])
            
            # Convert target to integer
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
            
            # Handle missing values in features
            train_data = self.handle_missing_values(train_data)
            test_data = self.handle_missing_values(test_data)
            
            # Encode categorical features
            train_data = self.encode_categorical_features(train_data)
            test_data = self.encode_categorical_features(test_data)
            
            # Drop target column and CustomerID before normalization
            columns_to_drop = [target_column, 'CustomerID']
            X_train = train_data.drop(columns=columns_to_drop)
            X_test = test_data.drop(columns=columns_to_drop)
            
            # Store feature columns for later use
            self.feature_columns = X_train.columns.tolist()
            
            # Normalize numerical features
            X_train = self.normalize_numerical_features(X_train)
            X_test = self.normalize_numerical_features(X_test)
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise Exception(f"Error in preprocessing pipeline: {str(e)}")
    
    def preprocess_single_customer(self, customer_data):
        """Preprocess a single customer's data for prediction"""
        try:
            if not isinstance(customer_data, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            if self.feature_columns is None:
                raise ValueError("Preprocessor has not been trained yet. Please preprocess training data first.")
            
            # Create a copy to avoid modifying the original data
            data = customer_data.copy()
            
            # If target column exists, drop it
            if 'Churn' in data.columns:
                data = data.drop(columns=['Churn'])
                
            # Drop CustomerID if it exists
            if 'CustomerID' in data.columns:
                data = data.drop(columns=['CustomerID'])
            
            # Ensure all required features are present
            missing_features = set(self.feature_columns) - set(data.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Reorder columns to match training data
            data = data[self.feature_columns]
            
            # Handle missing values
            data = self.handle_missing_values(data)
            
            # Encode categorical features
            data = self.encode_categorical_features(data)
            
            # Normalize numerical features
            data = self.normalize_numerical_features(data)
            
            return data
        except Exception as e:
            raise Exception(f"Error preprocessing single customer data: {str(e)}") 