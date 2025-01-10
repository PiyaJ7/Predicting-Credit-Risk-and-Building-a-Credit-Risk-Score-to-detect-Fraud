import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, f_oneway
from sklearn.preprocessing import LabelEncoder
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Load the datset
df = pd.read_csv('datasets/cleaned_fraud_detection_dataset.csv')

# Encode Charge_off_status
label_encoder = LabelEncoder()
df['Charge_off_status_encoded'] = label_encoder.fit_transform(df['charge_off_status'])

# Calculate Point Biserial Correlation for binary target
if df['Charge_off_status_encoded'].nunique() == 2:
    print("Point Biserial Correlation for Binary Target Variables:")
    for column in df.select_dtypes(include=[np.number]).columns:
        if column != 'Charge_off_status_encoded':
            corr, p_value = pointbiserialr(df[column], df['Charge_off_status_encoded'])
            print(f"Correlation between {column} and Charge_off_status: {corr:.3f}, p-value: {p_value:.3f}")

# Visualize the distribution of numerical features against Charge_off_status
output_folder = 'plots/EDA'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i, column in enumerate(df.select_dtypes(include=[np.number]).columns):
    if column != 'charge_off_status_encoded':
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='charge_off_status', y=column, data=df)
        plt.title(f"Distribution of {column} by Charge_off_status")
        
        plot_filename = os.path.join(output_folder, f"{column}_charge_off_status.png")
        plt.savefig(plot_filename)
        plt.close()

print(f"Plots saved to '{output_folder}' folder.")

# Add a constant to the dataset
X = add_constant(df[['number_of_delinquent_accounts', 'number_of_defaulted_accounts']])

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)