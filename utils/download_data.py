"""
Data download script for Telco Customer Churn dataset.
Developed by Kuldeep Choksi

Downloads the IBM Telco Customer Churn dataset from Kaggle.
Dataset URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""

import os
import pandas as pd
from pathlib import Path


def download_from_url():
    """
    Download dataset directly from source.
    
    The dataset is available at:
    https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
    """
    
    print("Downloading Telco Customer Churn dataset...")
    
    # Dataset URL (IBM's public GitHub repository)
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download
    try:
        df = pd.read_csv(url)
        
        # Save locally
        output_path = data_dir / "Telco-Customer-Churn.csv"
        df.to_csv(output_path, index=False)
        
        print(f"✓ Dataset downloaded successfully!")
        print(f"✓ Saved to: {output_path}")
        print(f"✓ Shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nAlternative: Download manually from Kaggle:")
        print("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        return None


def load_data():
    """
    Load dataset from local file if it exists.
    
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    
    data_path = Path("data/Telco-Customer-Churn.csv")
    
    if not data_path.exists():
        print("Dataset not found locally. Downloading...")
        return download_from_url()
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


if __name__ == "__main__":
    # Download or load dataset
    df = load_data()
    
    if df is not None:
        print("\n" + "="*60)
        print("DATASET INFORMATION")
        print("="*60)
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"\nColumn names:")
        for col in df.columns:
            print(f"  - {col}")
        
        print(f"\nTarget variable (Churn) distribution:")
        print(df['Churn'].value_counts())
        print(f"\nChurn rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.1f}%")