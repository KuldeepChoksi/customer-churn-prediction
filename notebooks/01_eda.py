"""
Exploratory Data Analysis for Telco Customer Churn
Developed by Kuldeep Choksi

Analyzes the dataset to understand:
- Feature distributions
- Churn patterns
- Correlations
- Missing values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create results directory
results_dir = Path("results/eda")
results_dir.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the dataset"""
    data_path = Path("data/Telco-Customer-Churn.csv")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def basic_info(df):
    """Display basic dataset information"""
    print("\n" + "="*60)
    print("BASIC DATASET INFORMATION")
    print("="*60)
    
    print(f"\nShape: {df.shape}")
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values!")
    else:
        print(missing[missing > 0])
    
    print(f"\nTarget variable distribution:")
    print(df['Churn'].value_counts())
    print(f"\nChurn rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.1f}%")


def analyze_numerical_features(df):
    """Analyze numerical features"""
    print("\n" + "="*60)
    print("NUMERICAL FEATURES ANALYSIS")
    print("="*60)
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Check TotalCharges data type issue
    print(f"\nTotalCharges data type: {df['TotalCharges'].dtype}")
    
    # Convert TotalCharges to numeric (it might be stored as string)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    print(f"\nNumerical features summary:")
    print(df[numerical_cols].describe())
    
    # Check for missing values after conversion
    print(f"\nMissing values in TotalCharges: {df['TotalCharges'].isnull().sum()}")
    
    # Visualize distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.savefig(results_dir / "numerical_distributions.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {results_dir / 'numerical_distributions.png'}")
    plt.close()
    
    return df


def analyze_churn_by_features(df):
    """Analyze churn rate by different features"""
    print("\n" + "="*60)
    print("CHURN ANALYSIS BY FEATURES")
    print("="*60)
    
    # Key categorical features to analyze
    cat_features = ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(cat_features):
        # Calculate churn rate by feature
        churn_rate = df.groupby(feature)['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).sort_values(ascending=False)
        
        print(f"\nChurn rate by {feature}:")
        print(churn_rate)
        
        # Visualize
        churn_rate.plot(kind='bar', ax=axes[idx], color='coral', edgecolor='black')
        axes[idx].set_title(f'Churn Rate by {feature}')
        axes[idx].set_ylabel('Churn Rate (%)')
        axes[idx].set_xlabel(feature)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "churn_by_features.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {results_dir / 'churn_by_features.png'}")
    plt.close()


def analyze_tenure_vs_churn(df):
    """Analyze relationship between tenure and churn"""
    print("\n" + "="*60)
    print("TENURE VS CHURN ANALYSIS")
    print("="*60)
    
    # Create tenure groups
    df['tenure_group'] = pd.cut(df['tenure'], 
                                bins=[0, 12, 24, 48, 72],
                                labels=['0-12 months', '12-24 months', 
                                       '24-48 months', '48-72 months'])
    
    # Churn rate by tenure group
    churn_by_tenure = df.groupby('tenure_group')['Churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    )
    
    print(f"\nChurn rate by tenure group:")
    print(churn_by_tenure)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    churn_by_tenure.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('Churn Rate by Tenure Group')
    axes[0].set_ylabel('Churn Rate (%)')
    axes[0].set_xlabel('Tenure Group')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Distribution of churned vs not churned customers by tenure
    df_churned = df[df['Churn'] == 'Yes']['tenure']
    df_not_churned = df[df['Churn'] == 'No']['tenure']
    
    axes[1].hist([df_not_churned, df_churned], bins=30, 
                label=['Not Churned', 'Churned'], 
                color=['green', 'red'], alpha=0.6, edgecolor='black')
    axes[1].set_title('Tenure Distribution by Churn Status')
    axes[1].set_xlabel('Tenure (months)')
    axes[1].set_ylabel('Number of Customers')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "tenure_vs_churn.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {results_dir / 'tenure_vs_churn.png'}")
    plt.close()


def analyze_charges_vs_churn(df):
    """Analyze relationship between charges and churn"""
    print("\n" + "="*60)
    print("CHARGES VS CHURN ANALYSIS")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Monthly charges
    df_churned = df[df['Churn'] == 'Yes']['MonthlyCharges']
    df_not_churned = df[df['Churn'] == 'No']['MonthlyCharges']
    
    axes[0].hist([df_not_churned, df_churned], bins=30,
                label=['Not Churned', 'Churned'],
                color=['green', 'red'], alpha=0.6, edgecolor='black')
    axes[0].set_title('Monthly Charges Distribution by Churn')
    axes[0].set_xlabel('Monthly Charges ($)')
    axes[0].set_ylabel('Number of Customers')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    print(f"\nAverage Monthly Charges:")
    print(f"  Not Churned: ${df_not_churned.mean():.2f}")
    print(f"  Churned: ${df_churned.mean():.2f}")
    
    # Total charges
    df_churned_total = df[df['Churn'] == 'Yes']['TotalCharges'].dropna()
    df_not_churned_total = df[df['Churn'] == 'No']['TotalCharges'].dropna()
    
    axes[1].hist([df_not_churned_total, df_churned_total], bins=30,
                label=['Not Churned', 'Churned'],
                color=['green', 'red'], alpha=0.6, edgecolor='black')
    axes[1].set_title('Total Charges Distribution by Churn')
    axes[1].set_xlabel('Total Charges ($)')
    axes[1].set_ylabel('Number of Customers')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    print(f"\nAverage Total Charges:")
    print(f"  Not Churned: ${df_not_churned_total.mean():.2f}")
    print(f"  Churned: ${df_churned_total.mean():.2f}")
    
    plt.tight_layout()
    plt.savefig(results_dir / "charges_vs_churn.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {results_dir / 'charges_vs_churn.png'}")
    plt.close()


def correlation_analysis(df):
    """Analyze correlations between features"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Select numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Convert Churn to binary
    df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)
    
    # Calculate correlations
    corr_data = df[numerical_features + ['Churn_binary']].corr()
    
    print(f"\nCorrelation with Churn:")
    print(corr_data['Churn_binary'].sort_values(ascending=False))
    
    # Visualize
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(results_dir / "correlation_matrix.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {results_dir / 'correlation_matrix.png'}")
    plt.close()


def key_insights(df):
    """Print key insights from the analysis"""
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    # 1. Contract type impact
    contract_churn = df.groupby('Contract')['Churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    )
    print(f"\n1. CONTRACT TYPE IMPACT:")
    print(f"   - Month-to-month: {contract_churn['Month-to-month']:.1f}% churn")
    print(f"   - One year: {contract_churn['One year']:.1f}% churn")
    print(f"   - Two year: {contract_churn['Two year']:.1f}% churn")
    print(f"   → Month-to-month contracts have {contract_churn['Month-to-month'] / contract_churn['Two year']:.1f}x higher churn!")
    
    # 2. Tenure impact
    new_customers = df[df['tenure'] <= 12]
    old_customers = df[df['tenure'] > 12]
    new_churn = (new_customers['Churn'] == 'Yes').sum() / len(new_customers) * 100
    old_churn = (old_customers['Churn'] == 'Yes').sum() / len(old_customers) * 100
    
    print(f"\n2. TENURE IMPACT:")
    print(f"   - New customers (≤12 months): {new_churn:.1f}% churn")
    print(f"   - Existing customers (>12 months): {old_churn:.1f}% churn")
    print(f"   → New customers churn {new_churn / old_churn:.1f}x more!")
    
    # 3. Tech support impact
    tech_support_churn = df.groupby('TechSupport')['Churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    )
    print(f"\n3. TECH SUPPORT IMPACT:")
    for support_type, churn_rate in tech_support_churn.items():
        print(f"   - {support_type}: {churn_rate:.1f}% churn")
    
    # 4. Payment method impact
    payment_churn = df.groupby('PaymentMethod')['Churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    ).sort_values(ascending=False)
    print(f"\n4. PAYMENT METHOD IMPACT:")
    for method, churn_rate in payment_churn.items():
        print(f"   - {method}: {churn_rate:.1f}% churn")


def main():
    """Run all EDA analyses"""
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Run analyses
    basic_info(df)
    df = analyze_numerical_features(df)
    analyze_churn_by_features(df)
    analyze_tenure_vs_churn(df)
    analyze_charges_vs_churn(df)
    correlation_analysis(df)
    key_insights(df)
    
    print("\n" + "="*60)
    print("✓ EDA COMPLETE!")
    print(f"✓ Visualizations saved to: {results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()