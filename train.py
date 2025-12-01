"""
Train customer churn prediction models.
Developed by Kuldeep Choksi

Trains and compares:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost (typically best)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_preprocessed_data():
    """Load preprocessed train/test data"""
    print("Loading preprocessed data...")
    data_dir = Path("data/processed")
    
    X_train = joblib.load(data_dir / 'X_train.pkl')
    X_test = joblib.load(data_dir / 'X_test.pkl')
    y_train = joblib.load(data_dir / 'y_train.pkl')
    y_test = joblib.load(data_dir / 'y_test.pkl')
    
    print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model.
    
    Simple baseline model.
    """
    print("\n" + "="*60)
    print("Training Logistic Regression...")
    print("="*60)
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    print("✓ Training complete")
    
    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest model.
    
    Good for feature importance and non-linear patterns.
    """
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    print("✓ Training complete")
    
    return model


def train_xgboost(X_train, y_train):
    """
    Train XGBoost model.
    
    Usually achieves best performance on tabular data.
    """
    print("\n" + "="*60)
    print("Training XGBoost...")
    print("="*60)
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    print("✓ Training complete")
    
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model and print metrics.
    
    Returns:
        dict: Dictionary of metrics
    """
    print(f"\nEvaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def compare_models(results):
    """
    Compare all models and visualize.
    
    Args:
        results: List of metric dictionaries
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('model')
    
    print("\n", df_results.round(4))
    
    # Find best model
    best_model = df_results['roc_auc'].idxmax()
    print(f"\n✓ Best Model: {best_model}")
    print(f"  ROC-AUC: {df_results.loc[best_model, 'roc_auc']:.4f}")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: All metrics
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    df_results[metrics_to_plot].plot(kind='bar', ax=axes[0], rot=0)
    axes[0].set_title('Model Performance Comparison')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc='lower right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: ROC-AUC focus
    df_results['roc_auc'].plot(kind='bar', ax=axes[1], color='coral', rot=0)
    axes[1].set_title('ROC-AUC Comparison')
    axes[1].set_ylabel('ROC-AUC Score')
    axes[1].set_ylim(0, 1)
    axes[1].axhline(y=0.5, color='red', linestyle='--', label='Random (0.5)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    results_dir = Path("results/models")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {results_dir / 'model_comparison.png'}")
    plt.close()
    
    return best_model


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """Plot confusion matrix for a model"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churn', 'Churn'],
                yticklabels=['Not Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.7, f'({cm_norm[i,j]:.1%})', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    results_dir = Path("results/models")
    plt.tight_layout()
    plt.savefig(results_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()


def save_best_model(model, model_name):
    """Save the best model"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(model, models_dir / f'best_model_{model_name.lower().replace(" ", "_")}.pkl')
    print(f"✓ Saved: {models_dir / f'best_model_{model_name.lower().replace(' ', '_')}.pkl'}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # Evaluate all models
    results = []
    results.append(evaluate_model(lr_model, X_test, y_test, "Logistic Regression"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest"))
    results.append(evaluate_model(xgb_model, X_test, y_test, "XGBoost"))
    
    # Compare models
    best_model_name = compare_models(results)
    
    # Plot confusion matrices
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(lr_model, X_test, y_test, "Logistic Regression")
    plot_confusion_matrix(rf_model, X_test, y_test, "Random Forest")
    plot_confusion_matrix(xgb_model, X_test, y_test, "XGBoost")
    
    # Save best model
    if best_model_name == "Logistic Regression":
        save_best_model(lr_model, best_model_name)
    elif best_model_name == "Random Forest":
        save_best_model(rf_model, best_model_name)
    else:
        save_best_model(xgb_model, best_model_name)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()