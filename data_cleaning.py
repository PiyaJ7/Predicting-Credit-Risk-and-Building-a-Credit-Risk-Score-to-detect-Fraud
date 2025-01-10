import pandas as pd
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

# Loading the CSV file
data = pd.read_csv('datasets/fraud_detection_dataset.csv')

# Preview the dataset
print(data.head())
print(data.info())

###################### 1. Handling Missing values ######################

#Identify missing values
print("Missing Values")
print(data.isnull().sum())


# Calculate missing value percentages
print()
print("Missing Values percentage: ")
missing_percent = (data.isnull().sum() / len(data)) * 100
print(missing_percent)

#fico_score - Left Skewed Distribution (using the median)
data['fico_score'].fillna(data['fico_score'].median(), inplace=True)

#Imputing with median to avoid being influenced by outliers
data['avg_balance_last_12months'].fillna(data['avg_balance_last_12months'].median(), inplace=True)

# Impute missing values with the mode
mode_value = data['number_of_delinquent_accounts'].mode()[0]
data['number_of_delinquent_accounts'].fillna(mode_value, inplace=True)

# Impute missing values with the mode
mode_value = data['unusual_submission_pattern'].mode()[0]
data['unusual_submission_pattern'].fillna(mode_value, inplace=True)

print(data.isnull().sum())

###################### 2. Fix Incorrect Data Types ######################

#Check the data types
print(data.dtypes)

# Convert date columns to datetime
data['account_open_date'] = pd.to_datetime(data['account_open_date'], errors='coerce')
data['earliest_credit_account'] = pd.to_datetime(data['earliest_credit_account'], errors='coerce')
data['recent_trade_activity'] = pd.to_datetime(data['recent_trade_activity'], errors='coerce')

print(data.dtypes)

###################### 3. Handle Outliers ######################

# Display descriptive statistics for numeric columns
print(data['debt_to_income_ratio'].describe())

# List of numerical columns
numerical_columns = [
    'age', 'income_level', 'fico_score', 'delinquency_status',
    'number_of_credit_applications', 'debt_to_income_ratio', 
    'max_balance', 'avg_balance_last_12months',
    'number_of_delinquent_accounts', 'number_of_defaulted_accounts',
    'new_accounts_opened_last_12months'
]

outliers_dict = {}

for column in numerical_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    outliers_dict[column] = outliers.index.tolist()
    
    print(f"{column} - Number of Outliers: {len(outliers)}")
    print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    print(outliers[column].head()) 

# Checking for invalid fico_score values - fico_score values should be between 300 and 850
invalid_fico_scores = data[(data['fico_score'] < 300) | (data['fico_score'] > 850)]
print(f"Number of invalid fico_score values: {len(invalid_fico_scores)}")

# Remove rows with invalid fico_score values
data = data[(data['fico_score'] >= 300) & (data['fico_score'] <= 850)]


outlier_columns = ['income_level', 'fico_score', 'delinquency_status', 
                     'number_of_credit_applications', 'debt_to_income_ratio', 'max_balance']

###################### 4. Handle duplicate rows ######################

# Checking duplicate rows
duplicate_rows = data[data.duplicated()]

# Number of duplicate rows
print(f"Number of duplicate rows: {duplicate_rows.shape[0]}") # There is no any duplicate rows

# Saving cleaned dataset
data.to_csv('datasets/cleaned_fraud_detection_dataset.csv', index=False)