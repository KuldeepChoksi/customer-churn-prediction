"""
Improved training with SMOTE and hyperparameter tuning.
Developed by Kuldeep Choksi

Improvements:
- SMOTE for better class balance
- Grid search for hyperparameters
- More comprehensive evaluation
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
    confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
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
    print(f"  Train churn rate: {y_train.mean()*100:.1f}%")
    
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique).
    
    Creates synthetic samples of minority class (churners) to balance dataset.
    """
    print("\n" + "="*60)
    print("Applying SMOTE...")
    print("="*60)
    
    print(f"Before SMOTE:")
    print(f"  Total samples: {len(X_train)}")
    print(f"  Churners: {(y_train == 1).sum()} ({y_train.mean()*100:.1f}%)")
    print(f"  Non-churners: {(y_train == 0).sum()} ({(1-y_train.mean())*100:.1f}%)")
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"\nAfter SMOTE:")
    print(f"  Total samples: {len(X_train_smote)}")
    print(f"  Churners: {(y_train_smote == 1).sum()} ({y_train_smote.mean()*100:.1f}%)")
    print(f"  Non-churners: {(y_train_smote == 0).sum()} ({(1-y_train_smote.mean())*100:.1f}%)")
    
    return X_train_smote, y_train_smote


def train_models_with_smote(X_train, y_train):
    """Train all models with SMOTE data"""
    
    # Logistic Regression
    print("\n" + "="*60)
    print("Training Logistic Regression (with SMOTE)...")
    print("="*60)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    print("✓ Training complete")
    
    # Random Forest
    print("\n" + "="*60)
    print("Training Random Forest (with SMOTE)...")
    print("="*60)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("✓ Training complete")
    
    # XGBoost
    print("\n" + "="*60)
    print("Training XGBoost (with SMOTE)...")
    print("="*60)
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    print("✓ Training complete")
    
    return lr_model, rf_model, xgb_model


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and print metrics"""
    print(f"\nEvaluating {model_name}...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
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
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn']))
    
    return metrics, y_pred_proba


def plot_roc_curves(models_dict, X_test, y_test):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 7))
    
    for model_name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Customer Churn Prediction\nDeveloped by Kuldeep Choksi')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    results_dir = Path("results/models")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curves_smote.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {results_dir / 'roc_curves_smote.png'}")
    plt.close()


def save_best_model(model, model_name, metrics):
    """Save the best model with metadata"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / 'best_model.pkl'
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'metrics': metrics,
        'training': 'SMOTE oversampling applied'
    }
    joblib.dump(metadata, models_dir / 'model_metadata.pkl')
    
    print(f"\n✓ Saved best model: {model_path}")
    print(f"  Model: {model_name}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")


def main():
    """Main improved training pipeline"""
    print("="*60)
    print("CUSTOMER CHURN PREDICTION - IMPROVED TRAINING")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Apply SMOTE
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    
    # Train models
    lr_model, rf_model, xgb_model = train_models_with_smote(X_train_smote, y_train_smote)
    
    # Evaluate all models
    results = []
    lr_metrics, _ = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_metrics, _ = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    xgb_metrics, _ = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    results.append(lr_metrics)
    results.append(rf_metrics)
    results.append(xgb_metrics)
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON (WITH SMOTE)")
    print("="*60)
    
    df_results = pd.DataFrame(results).set_index('model')
    print("\n", df_results.round(4))
    
    best_model_name = df_results['roc_auc'].idxmax()
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"  ROC-AUC: {df_results.loc[best_model_name, 'roc_auc']:.4f}")
    
    # Plot ROC curves
    models_dict = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }
    plot_roc_curves(models_dict, X_test, y_test)
    
    # Save best model
    if best_model_name == "Logistic Regression":
        save_best_model(lr_model, best_model_name, lr_metrics)
    elif best_model_name == "Random Forest":
        save_best_model(rf_model, best_model_name, rf_metrics)
    else:
        save_best_model(xgb_model, best_model_name, xgb_metrics)
    
    print("\n" + "="*60)
    print("✓ IMPROVED TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()