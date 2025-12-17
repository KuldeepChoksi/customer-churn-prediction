"""
Additional metrics utilities for churn prediction.
Developed by Kuldeep Choksi
"""

def calculate_customer_lifetime_value(monthly_charges, tenure_months):
    """Calculate estimated customer lifetime value"""
    return monthly_charges * tenure_months

def calculate_churn_cost(num_customers, avg_lifetime_value):
    """Calculate total cost of churn"""
    return num_customers * avg_lifetime_value
