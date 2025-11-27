"""
Data preprocessing and feature engineering for churn prediction.
Developed by Kuldeep Choksi

Handles:
- Missing value imputation
- Feature encoding
- Feature engineering
- Train/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import joblib


def load_and_clean_data(data_path='data/Telco-Customer-Churn.csv'):
    """
    Load dataset and handle basic cleaning.
    
    Returns:
        pandas.DataFrame: Cleaned dataset
    """
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Convert TotalCharges to numeric (stored as string with spaces)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Handle missing values in TotalCharges (11 customers with 0 tenure)
    # These are new customers, so TotalCharges = MonthlyCharges
    print(f"\nMissing values in TotalCharges: {df['TotalCharges'].isnull().sum()}")
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    
    print(f"✓ Loaded {len(df)} customers")
    return df


def feature_engineering(df):
    """
    Create new features from existing ones.
    
    Args:
        df: Input dataframe
    
    Returns:
        pandas.DataFrame: Dataframe with new features
    """
    print("\nFeature Engineering...")
    
    df = df.copy()
    
    # 1. Tenure groups (based on EDA insights)
    df['tenure_group'] = pd.cut(df['tenure'], 
                                bins=[0, 12, 24, 48, 73],
                                labels=['0-12', '12-24', '24-48', '48+'])
    
    # 2. Average monthly spend (total / tenure, handle 0 tenure)
    df['avg_monthly_spend'] = df.apply(
        lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 
        else row['MonthlyCharges'], 
        axis=1
    )
    
    # 3. Services count (how many services customer has)
    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                   'StreamingTV', 'StreamingMovies']
    
    df['services_count'] = df[service_cols].apply(
        lambda x: sum((x != 'No') & (x != 'No internet service')), axis=1
    )
    
    # 4. Has support (tech support or online security)
    df['has_support'] = ((df['TechSupport'] == 'Yes') | 
                         (df['OnlineSecurity'] == 'Yes')).astype(int)
    
    # 5. Is new customer (first year critical based on EDA)
    df['is_new_customer'] = (df['tenure'] <= 12).astype(int)
    
    # 6. High monthly charges (above median)
    median_charges = df['MonthlyCharges'].median()
    df['high_charges'] = (df['MonthlyCharges'] > median_charges).astype(int)
    
    print(f"✓ Created 6 new features")
    return df


def encode_features(df, is_training=True, encoders=None):
    """
    Encode categorical variables.
    
    Args:
        df: Input dataframe
        is_training: Whether this is training data (fit encoders) or test (use existing)
        encoders: Dictionary of fitted encoders (for test data)
    
    Returns:
        tuple: (encoded_df, encoders_dict)
    """
    print("\nEncoding categorical features...")
    
    df = df.copy()
    
    # Binary features (Yes/No) -> 1/0
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)
    
    # Target variable
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # Multi-class categorical features -> one-hot encoding
    categorical_cols = ['gender', 'MultipleLines', 'InternetService', 
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 
                       'Contract', 'PaymentMethod', 'tenure_group']
    
    # One-hot encode
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"✓ Encoded features, total columns: {len(df.columns)}")
    
    return df, None


def prepare_train_test_split(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Args:
        df: Input dataframe
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("\nSplitting data...")
    
    # Remove customerID (not a feature)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    print(f"✓ Train churn rate: {y_train.mean()*100:.1f}%")
    print(f"✓ Test churn rate: {y_test.mean()*100:.1f}%")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale numerical features to have mean=0, std=1.
    
    Args:
        X_train: Training features
        X_test: Test features
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print("\nScaling features...")
    
    # Identify numerical columns (not binary 0/1)
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                     'avg_monthly_spend', 'services_count']
    
    # Only scale if columns exist
    numerical_cols = [col for col in numerical_cols if col in X_train.columns]
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"✓ Scaled {len(numerical_cols)} numerical features")
    
    return X_train_scaled, X_test_scaled, scaler


def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler):
    """
    Save preprocessed data and scaler for later use.
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        scaler: Fitted StandardScaler
    """
    print("\nSaving preprocessed data...")
    
    data_dir = Path("data/processed")
    data_dir.mkdir(exist_ok=True)
    
    # Save as pickle for exact reproducibility
    joblib.dump(X_train, data_dir / 'X_train.pkl')
    joblib.dump(X_test, data_dir / 'X_test.pkl')
    joblib.dump(y_train, data_dir / 'y_train.pkl')
    joblib.dump(y_test, data_dir / 'y_test.pkl')
    joblib.dump(scaler, data_dir / 'scaler.pkl')
    
    print(f"✓ Saved to {data_dir}/")


def load_preprocessed_data():
    """
    Load preprocessed data.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    data_dir = Path("data/processed")
    
    X_train = joblib.load(data_dir / 'X_train.pkl')
    X_test = joblib.load(data_dir / 'X_test.pkl')
    y_train = joblib.load(data_dir / 'y_train.pkl')
    y_test = joblib.load(data_dir / 'y_test.pkl')
    scaler = joblib.load(data_dir / 'scaler.pkl')
    
    print(f"✓ Loaded preprocessed data from {data_dir}/")
    return X_train, X_test, y_train, y_test, scaler


def preprocess_pipeline():
    """
    Complete preprocessing pipeline.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    print("="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    # 1. Load and clean
    df = load_and_clean_data()
    
    # 2. Feature engineering
    df = feature_engineering(df)
    
    # 3. Encode categorical features
    df, _ = encode_features(df)
    
    # 4. Train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    # 5. Scale features
    X_train, X_test, scaler = scale_features(X_train, X_test)
    
    # 6. Save
    save_preprocessed_data(X_train, X_test, y_train, y_test, scaler)
    
    print("\n" + "="*60)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*60)
    
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Run the pipeline
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline()
    
    print(f"\nFinal dataset shape:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  Features: {X_train.shape[1]}")