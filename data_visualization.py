import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

output_dir = 'plots/Data visualization'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv('datasets/fraud_detection_dataset.csv')

# 1. Histogram for Age
plt.figure(figsize=(8, 6))
sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
plt.title('Histogram of Age', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig(f'{output_dir}/histogram_age.png')
plt.close()

# 2. Bar plot for location
location_counts = df['location'].value_counts()
colors = plt.cm.viridis(np.linspace(0, 1, len(location_counts)))

plt.figure(figsize=(12, 8))
bars = location_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Count of Locations', fontsize=16)
plt.xlabel('Location', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.ylim(625, 750)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{output_dir}/location_counts_bar_chart.png')
plt.close()

# 3. Bar chart for Occupation
plt.figure(figsize=(10, 6))
df['occupation'].value_counts().plot(kind='bar', color='teal')
plt.title('Bar Chart of Occupations', fontsize=16)
plt.xlabel('Occupation', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/bar_chart_occupation.png')
plt.close()

# 4. Density plot for Income Level
plt.figure(figsize=(8, 6))
sns.kdeplot(df['income_level'], fill=True, color='purple')
plt.title('Density Plot of Income Level', fontsize=16)
plt.xlabel('Income Level', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.savefig(f'{output_dir}/density_income_level.png')
plt.close()

# 5. Histogram for FICO Score
plt.figure(figsize=(8, 6))
sns.histplot(df['fico_score'], bins=30, kde=True, color='gold')
plt.title('Histogram of FICO Score', fontsize=16)
plt.xlabel('FICO Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig(f'{output_dir}/histogram_fico_score.png')
plt.close()

# 6. Pie chart for Charge Off Status
charge_off_counts = df['charge_off_status'].value_counts()

plt.figure(figsize=(8, 8))
charge_off_counts.plot(kind='pie', autopct='%1.1f%%', colors=['orange', 'lightblue'], legend=True)
plt.title('Charge Off Status', fontsize=16)
plt.ylabel('')
plt.tight_layout()
plt.savefig(f'{output_dir}/pie_charge_off_status.png')
plt.close()

# 7. Bar chart for Number of Credit Applications
plt.figure(figsize=(8, 6))
sns.barplot(x=df['number_of_credit_applications'].value_counts().index, 
            y=df['number_of_credit_applications'].value_counts().values, palette='viridis')
plt.title('Bar Chart of Credit Applications', fontsize=16)
plt.xlabel('Number of Applications', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/bar_chart_credit_applications.png')
plt.close()

# 8. Correlation Heatmap
numeric_data = df.select_dtypes(include=['number', 'float', 'int'])
if 'date_column' in df.columns:
    df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')

data_cleaned = numeric_data.dropna()

plt.figure(figsize=(12, 10))
correlation_matrix = data_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Variables', fontsize=16)
plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_heatmap.png')
plt.close()

# 9. Density plot for Debt to Income Ratio
plt.figure(figsize=(8, 6))
sns.kdeplot(df['debt_to_income_ratio'], fill=True, color='green')
plt.title('Density Plot of Debt to Income Ratio', fontsize=16)
plt.xlabel('Debt to Income Ratio', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.savefig(f'{output_dir}/density_debt_to_income_ratio.png')
plt.close()

# 10. Pie chart for Payment Methods High Risk
payment_methods_counts = df['payment_methods_high_risk'].value_counts()
plt.figure(figsize=(8, 8))
payment_methods_counts.plot(kind='pie', autopct='%1.1f%%', colors=['pink', 'cyan'], legend=True)
plt.title('Payment Methods High Risk', fontsize=16)
plt.ylabel('')
plt.tight_layout()
plt.savefig(f'{output_dir}/pie_payment_methods_high_risk.png')
plt.close()

# 11. Density plot for Max Balance
plt.figure(figsize=(8, 6))
sns.kdeplot(df['max_balance'], fill=True, color='orange')
plt.title('Density Plot of Max Balance', fontsize=16)
plt.xlabel('Max Balance', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.savefig(f'{output_dir}/density_plot_for_max_balance.png')
plt.close()

# 12. Density plot for Avg Balance Last 12 Months
plt.figure(figsize=(8, 6))
sns.kdeplot(df['avg_balance_last_12months'], fill=True, color='blue')
plt.title('Density Plot of Avg Balance (Last 12 Months)', fontsize=16)
plt.xlabel('Avg Balance', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.savefig(f'{output_dir}/density_plot_for_Avg_Balance_Last_12_Months.png')
plt.close()

# 13. Bar chart for Number of Delinquent Accounts
plt.figure(figsize=(8, 6))
sns.barplot(x=df['number_of_delinquent_accounts'].value_counts().index, 
            y=df['number_of_delinquent_accounts'].value_counts().values, palette='magma')
plt.title('Bar Chart of Delinquent Accounts', fontsize=16)
plt.xlabel('Number of Delinquent Accounts', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/bar_chart_for_Number_of_Delinquent_Accounts.png')
plt.close()

# 14. Bar chart for Number of Defaulted Accounts
plt.figure(figsize=(8, 6))
sns.barplot(x=df['number_of_defaulted_accounts'].value_counts().index, 
            y=df['number_of_defaulted_accounts'].value_counts().values, palette='plasma')
plt.title('Bar Chart of Defaulted Accounts', fontsize=16)
plt.xlabel('Number of Defaulted Accounts', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/bar_chart_for_Number_of_Defaulted_Accounts.png')
plt.close()

# 15. Bar chart for New Accounts Opened Last 12 Months
plt.figure(figsize=(8, 6))
sns.barplot(x=df['new_accounts_opened_last_12months'].value_counts().index, 
            y=df['new_accounts_opened_last_12months'].value_counts().values, palette='cividis')
plt.title('Bar Chart of New Accounts (Last 12 Months)', fontsize=16)
plt.xlabel('Number of New Accounts', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/bar_chart_for_New_Accounts_Opened_Last_12_Months.png')
plt.close()

# 16. Pie charts for Boolean Flags
boolean_flags = ['public_records_flag', 'watchlist_blacklist_flag', 'applications_submitted_during_odd_hours', 
                 'unusual_submission_pattern', 'multiple_applications_short_time_period']

for flag in boolean_flags:
    plt.figure(figsize=(8, 8))
    df[flag].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=['True', 'False'], colors=['gold', 'lightgreen'])
    plt.title(f'{flag.replace("_", " ").title()}', fontsize=16)
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pie_charts_for_Boolean_Flags.png')
    plt.close()