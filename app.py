"""
Customer Churn Prediction - Interactive Demo
Developed by Kuldeep Choksi

Try the model live! Upload customer data or enter manually to predict churn risk.
"""

import gradio as gr
import pandas as pd
import joblib
import numpy as np
from pathlib import Path


# Load model and scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('data/processed/scaler.pkl')
metadata = joblib.load('models/model_metadata.pkl')


def predict_churn(
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method,
    monthly_charges, total_charges
):
    """
    Predict churn probability for a customer.
    
    Returns risk level, probability, and recommendations.
    """
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    # Feature engineering (same as training)
    input_data['tenure_group'] = pd.cut(input_data['tenure'], 
                                        bins=[0, 12, 24, 48, 73],
                                        labels=['0-12', '12-24', '24-48', '48+'])
    
    input_data['avg_monthly_spend'] = input_data['TotalCharges'] / input_data['tenure'] if tenure > 0 else monthly_charges
    
    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                   'StreamingTV', 'StreamingMovies']
    input_data['services_count'] = sum((input_data[col].iloc[0] != 'No' and 
                                        input_data[col].iloc[0] != 'No internet service') 
                                       for col in service_cols)
    
    input_data['has_support'] = int((input_data['TechSupport'].iloc[0] == 'Yes') or 
                                    (input_data['OnlineSecurity'].iloc[0] == 'Yes'))
    input_data['is_new_customer'] = int(tenure <= 12)
    input_data['high_charges'] = int(monthly_charges > 70)
    
    # Encode (same as training)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        input_data[col] = (input_data[col] == 'Yes').astype(int)
    
    categorical_cols = ['gender', 'MultipleLines', 'InternetService', 
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 
                       'Contract', 'PaymentMethod', 'tenure_group']
    
    input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)
    
    # Ensure all columns from training are present
    # Load training columns
    X_train = joblib.load('data/processed/X_train.pkl')
    for col in X_train.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder columns to match training
    input_data = input_data[X_train.columns]
    
    # Predict
    churn_prob = model.predict_proba(input_data)[0, 1]
    churn_pred = model.predict(input_data)[0]
    
    # Risk level
    if churn_prob < 0.3:
        risk_level = "🟢 LOW RISK"
        risk_color = "green"
    elif churn_prob < 0.6:
        risk_level = "🟡 MEDIUM RISK"
        risk_color = "orange"
    else:
        risk_level = "🔴 HIGH RISK"
        risk_color = "red"
    
    # Recommendations
    recommendations = []
    
    if contract == "Month-to-month":
        recommendations.append("• Offer annual contract with 15% discount (month-to-month customers churn at 42.7% vs 2.8% for 2-year contracts)")
    
    if tenure <= 12:
        recommendations.append("• Assign dedicated customer success manager (new customers churn at 47.4% vs 17.1% for existing)")
    
    if tech_support == "No" and internet_service != "No":
        recommendations.append("• Provide free tech support trial (customers without support churn at 41.6% vs 15.2% with support)")
    
    if payment_method == "Electronic check":
        recommendations.append("• Incentivize automatic payment method switch (electronic check users churn at 45.3% vs 15-17% for automatic)")
    
    if monthly_charges > 70:
        recommendations.append("• Review pricing plan for value optimization (churned customers pay $74.44/mo vs $61.27 for retained)")
    
    if not recommendations:
        recommendations.append("• Customer appears stable - maintain current service quality")
    
    # Format output
    result = f"""
### Prediction: {'WILL CHURN' if churn_pred == 1 else 'WILL NOT CHURN'}

**Churn Probability:** {churn_prob:.1%}

**Risk Level:** {risk_level}

---

### Retention Recommendations

{chr(10).join(recommendations)}

---

### Model Information
- Model: {metadata['model_name']}
- ROC-AUC: {metadata['metrics']['roc_auc']:.3f}
- Training: {metadata['training']}
    """
    
    return result


# Gradio Interface
with gr.Blocks(title="Customer Churn Prediction") as demo:
    gr.Markdown("""
    # Customer Churn Prediction
    ### Developed by Kuldeep Choksi
    
    Predict customer churn risk using machine learning. Enter customer details below to get:
    - Churn probability
    - Risk level assessment  
    - Personalized retention recommendations
    
    **Model:** Random Forest with SMOTE (0.833 ROC-AUC)
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Customer Demographics")
            gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
            senior_citizen = gr.Radio(["Yes", "No"], label="Senior Citizen", value="No")
            partner = gr.Radio(["Yes", "No"], label="Has Partner", value="No")
            dependents = gr.Radio(["Yes", "No"], label="Has Dependents", value="No")
            tenure = gr.Slider(0, 72, value=12, step=1, label="Tenure (months)")
            
        with gr.Column():
            gr.Markdown("### Services")
            phone_service = gr.Radio(["Yes", "No"], label="Phone Service", value="Yes")
            multiple_lines = gr.Radio(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")
            internet_service = gr.Radio(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
            online_security = gr.Radio(["Yes", "No", "No internet service"], label="Online Security", value="No")
            online_backup = gr.Radio(["Yes", "No", "No internet service"], label="Online Backup", value="No")
            device_protection = gr.Radio(["Yes", "No", "No internet service"], label="Device Protection", value="No")
            tech_support = gr.Radio(["Yes", "No", "No internet service"], label="Tech Support", value="No")
            streaming_tv = gr.Radio(["Yes", "No", "No internet service"], label="Streaming TV", value="No")
            streaming_movies = gr.Radio(["Yes", "No", "No internet service"], label="Streaming Movies", value="No")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Account Information")
            contract = gr.Radio(["Month-to-month", "One year", "Two year"], label="Contract Type", value="Month-to-month")
            paperless_billing = gr.Radio(["Yes", "No"], label="Paperless Billing", value="Yes")
            payment_method = gr.Radio(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                label="Payment Method",
                value="Electronic check"
            )
            monthly_charges = gr.Slider(18, 120, value=70, step=0.1, label="Monthly Charges ($)")
            total_charges = gr.Slider(18, 9000, value=840, step=1, label="Total Charges ($)")
    
    predict_btn = gr.Button("Predict Churn Risk", variant="primary")
    
    output = gr.Markdown(label="Prediction Results")
    
    predict_btn.click(
        fn=predict_churn,
        inputs=[
            gender, senior_citizen, partner, dependents, tenure,
            phone_service, multiple_lines, internet_service,
            online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies,
            contract, paperless_billing, payment_method,
            monthly_charges, total_charges
        ],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### About This Model
    
    This churn prediction model was trained on the IBM Telco Customer Churn dataset using Random Forest with SMOTE oversampling.
    
    **Performance:**
    - Accuracy: 76.0%
    - Precision: 54% (minimize false alarms)
    - Recall: 68% (catch actual churners)
    - ROC-AUC: 0.833
    
    **Business Value:**
    Based on industry research showing customer acquisition costs 5-25x more than retention (Harvard Business Review), 
    this model enables proactive retention campaigns with an estimated 257% ROI.
    
    **References:**
    - Gallo, A. (2014). "The Value of Keeping the Right Customers." Harvard Business Review
    - IBM Telco Customer Churn Dataset
    
    **GitHub:** [View source code](https://github.com/KuldeepChoksi/customer-churn-prediction)
    """)


if __name__ == "__main__":
    demo.launch(share=False)