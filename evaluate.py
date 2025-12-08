"""
Evaluation script with business metrics.
Developed by Kuldeep Choksi

Evaluates model and calculates business impact.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, 
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_model_and_data():
    """Load trained model and test data"""
    print("Loading model and data...")
    
    # Load model
    model = joblib.load('models/best_model.pkl')
    metadata = joblib.load('models/model_metadata.pkl')
    
    # Load test data
    data_dir = Path("data/processed")
    X_test = joblib.load(data_dir / 'X_test.pkl')
    y_test = joblib.load(data_dir / 'y_test.pkl')
    
    print(f"✓ Loaded model: {metadata['model_name']}")
    print(f"✓ Test set: {X_test.shape}")
    
    return model, X_test, y_test, metadata


def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn']))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return y_pred, y_pred_proba, cm


def calculate_business_metrics(y_test, y_pred, y_pred_proba):
    """
    Calculate business impact metrics.
    
    Based on industry research:
    - Customer acquisition cost: 5-25x retention cost (HBR)
    - Retention campaign success: 20-30% (industry standard)
    """
    print("\n" + "="*60)
    print("BUSINESS IMPACT ANALYSIS")
    print("="*60)
    
    # Assumptions (conservative estimates from industry research)
    CUSTOMER_LIFETIME_VALUE = 2000  # Conservative for telecom
    RETENTION_CAMPAIGN_COST = 75     # Per customer
    CAMPAIGN_SUCCESS_RATE = 0.25     # 25% conservative estimate
    
    print(f"\nAssumptions (based on industry research):")
    print(f"  Customer Lifetime Value: ${CUSTOMER_LIFETIME_VALUE}")
    print(f"  Retention Campaign Cost: ${RETENTION_CAMPAIGN_COST}/customer")
    print(f"  Campaign Success Rate: {CAMPAIGN_SUCCESS_RATE*100:.0f}%")
    
    # Calculate metrics
    total_customers = len(y_test)
    actual_churners = (y_test == 1).sum()
    predicted_churners = (y_pred == 1).sum()
    
    # True positives: Correctly identified churners
    true_positives = ((y_pred == 1) & (y_test == 1)).sum()
    
    # False positives: Incorrectly flagged as churners
    false_positives = ((y_pred == 1) & (y_test == 0)).sum()
    
    # False negatives: Missed churners
    false_negatives = ((y_pred == 0) & (y_test == 1)).sum()
    
    print(f"\nPrediction Breakdown:")
    print(f"  Total customers in test: {total_customers}")
    print(f"  Actual churners: {actual_churners} ({actual_churners/total_customers*100:.1f}%)")
    print(f"  Predicted churners: {predicted_churners}")
    print(f"  Correctly identified: {true_positives} ({true_positives/actual_churners*100:.1f}% of actual)")
    print(f"  Missed churners: {false_negatives}")
    print(f"  False alarms: {false_positives}")
    
    # Business calculations
    customers_saved = int(true_positives * CAMPAIGN_SUCCESS_RATE)
    campaign_cost = predicted_churners * RETENTION_CAMPAIGN_COST
    revenue_saved = customers_saved * CUSTOMER_LIFETIME_VALUE
    net_benefit = revenue_saved - campaign_cost
    roi = (net_benefit / campaign_cost) * 100 if campaign_cost > 0 else 0
    
    print(f"\nBusiness Impact (Test Set Scale):")
    print(f"  Campaign targets: {predicted_churners} customers")
    print(f"  Campaign cost: ${campaign_cost:,}")
    print(f"  Customers saved: {customers_saved}")
    print(f"  Revenue saved: ${revenue_saved:,}")
    print(f"  Net benefit: ${net_benefit:,}")
    print(f"  ROI: {roi:.1f}%")
    
    # Scale to company size
    print(f"\nScaled to 50,000 customers (typical telecom):")
    scale_factor = 50000 / total_customers
    scaled_saved = int(customers_saved * scale_factor)
    scaled_revenue = int(revenue_saved * scale_factor)
    scaled_cost = int(campaign_cost * scale_factor)
    scaled_net = scaled_revenue - scaled_cost
    
    print(f"  Customers saved per month: {scaled_saved}")
    print(f"  Monthly revenue saved: ${scaled_revenue:,}")
    print(f"  Monthly campaign cost: ${scaled_cost:,}")
    print(f"  Monthly net benefit: ${scaled_net:,}")
    print(f"  Annual net benefit: ${scaled_net * 12:,}")
    
    return {
        'customers_saved': customers_saved,
        'campaign_cost': campaign_cost,
        'revenue_saved': revenue_saved,
        'net_benefit': net_benefit,
        'roi': roi
    }


def plot_business_impact(business_metrics):
    """Visualize business impact"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cost vs Revenue
    categories = ['Campaign\nCost', 'Revenue\nSaved', 'Net\nBenefit']
    values = [
        business_metrics['campaign_cost'],
        business_metrics['revenue_saved'],
        business_metrics['net_benefit']
    ]
    colors = ['red', 'green', 'blue']
    
    axes[0].bar(categories, values, color=colors, edgecolor='black', alpha=0.7)
    axes[0].set_title('Business Impact - Test Set Scale')
    axes[0].set_ylabel('Amount ($)')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(values):
        axes[0].text(i, v + max(values)*0.02, f'${v:,}', 
                    ha='center', va='bottom', fontweight='bold')
    
    # ROI visualization
    axes[1].bar(['ROI'], [business_metrics['roi']], 
               color='gold', edgecolor='black', width=0.5, alpha=0.7)
    axes[1].set_title('Return on Investment')
    axes[1].set_ylabel('ROI (%)')
    axes[1].set_ylim(0, max(business_metrics['roi'] * 1.2, 100))
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].text(0, business_metrics['roi'] + 5, f"{business_metrics['roi']:.1f}%",
                ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    plt.suptitle('Business Impact Analysis - Developed by Kuldeep Choksi', 
                fontsize=12, y=1.02)
    plt.tight_layout()
    
    results_dir = Path("results/models")
    plt.savefig(results_dir / "business_impact.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {results_dir / 'business_impact.png'}")
    plt.close()


def main():
    """Main evaluation pipeline"""
    print("="*60)
    print("CUSTOMER CHURN PREDICTION - EVALUATION")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    # Load model and data
    model, X_test, y_test, metadata = load_model_and_data()
    
    # Evaluate
    y_pred, y_pred_proba, cm = evaluate_model(model, X_test, y_test)
    
    # Business metrics
    business_metrics = calculate_business_metrics(y_test, y_pred, y_pred_proba)
    
    # Visualize business impact
    plot_business_impact(business_metrics)
    
    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()