import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pointbiserialr, f_oneway

current_date = datetime.now()

# Load the dataset
df = pd.read_csv('datasets/cleaned_fraud_detection_dataset.csv')

# Account Age
df['derived_account_age'] = (pd.to_datetime(current_date) - pd.to_datetime(df['account_open_date'])).dt.days // 365

# Credit History Length
df['derived_credit_history_length'] = (pd.to_datetime(current_date) - pd.to_datetime(df['earliest_credit_account'])).dt.days // 365

# Recency of Activity
df['derived_recency_of_activity'] = (pd.to_datetime(current_date) - pd.to_datetime(df['recent_trade_activity'])).dt.days

# Delinquent-to-Credit Application Ratio
df['derived_delinquent_to_credit_ratio'] = df['number_of_delinquent_accounts'] / (df['number_of_credit_applications'] + 1)

# Default-to-Credit Application Ratio
df['derived_default_to_credit_ratio'] = df['number_of_defaulted_accounts'] / (df['number_of_credit_applications'] + 1)

# Balance Stability
df['derived_balance_stability'] = df['avg_balance_last_12months'] / (df['max_balance'] + 1)

# High Application Frequency
df['derived_high_application_frequency'] = ((df['multiple_applications_short_time_period'] == True) & (df['number_of_credit_applications'] > 5)).astype(int)

# Irregular Behavior Indicator
df['derived_behavioral_risk_flag'] = (
    df['unusual_submission_pattern'].astype(int) +
    df['applications_submitted_during_odd_hours'].astype(int) +
    df['watchlist_blacklist_flag'].astype(int)
)

# Income-to-Balance Ratio
df['derived_income_to_balance_ratio'] = df['income_level'] / (df['avg_balance_last_12months'] + 1)

# Debt-to-Income-to-Balance Ratio
df['derived_debt_income_balance_ratio'] = (
    (df['debt_to_income_ratio'] * df['avg_balance_last_12months']) / (df['income_level'] + 1)
)

# FICO Score with Delinquency
df['derived_fico_with_delinquency'] = df['fico_score'] / (1 + df['delinquency_status'])

# Late Payment Risk
df['derived_late_payment_risk'] = df['delinquency_status'] / (df['debt_to_income_ratio'] + 1)

# New Account Risk
df['derived_new_account_risk'] = df['new_accounts_opened_last_12months'] / (df['derived_credit_history_length'] + 1)

# Log Transform for Skewed Features
df['derived_log_debt_to_income_ratio'] = np.log1p(df['debt_to_income_ratio'])
df['derived_log_max_balance'] = np.log1p(df['max_balance'])
df['derived_log_avg_balance'] = np.log1p(df['avg_balance_last_12months'])

# Binning FICO Score
bins = [300, 579, 669, 739, 799, 850]
labels = ['Low', 'Fair', 'Good', 'Very Good', 'Excellent']
df['derived_fico_binned'] = pd.cut(df['fico_score'], bins=bins, labels=labels, right=True)

# Standardization/Normalization
df['derived_standardized_dti_ratio'] = (df['debt_to_income_ratio'] - df['debt_to_income_ratio'].mean()) / df['debt_to_income_ratio'].std()

# High-Risk Indicator
df['derived_high_risk_indicator'] = ((df['public_records_flag'] == True) & (df['watchlist_blacklist_flag'] == True) & (df['debt_to_income_ratio'] > 0.5)).astype(int)

# Combined Behavior Flag
df['derived_combined_behavior_flag'] = (
    df['unusual_submission_pattern'].astype(int) +
    df['applications_submitted_during_odd_hours'].astype(int) +
    df['watchlist_blacklist_flag'].astype(int)
)

# Save the dataset
df.head()
df.to_csv('datasets/feature_extraction.csv', index=False)

# Load the dataset after extract the features
extracted_data = pd.read_csv('datasets/feature_extraction.csv')

label_encoder = LabelEncoder()
extracted_data['Charge_off_status_encoded'] = label_encoder.fit_transform(df['charge_off_status'])

# Calculate Point Biserial Correlation for binary target
if extracted_data['Charge_off_status_encoded'].nunique() == 2:
    print("Point Biserial Correlation for Binary Target Variables:")
    for column in extracted_data.select_dtypes(include=[np.number]).columns:
        if column != 'Charge_off_status_encoded':
            corr, p_value = pointbiserialr(extracted_data[column], extracted_data['Charge_off_status_encoded'])
            print(f"Correlation between {column} and Charge_off_status: {corr:.3f}, p-value: {p_value:.3f}")