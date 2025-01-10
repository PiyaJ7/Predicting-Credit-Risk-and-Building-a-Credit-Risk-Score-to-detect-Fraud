from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Load the dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Fill missing values
    df.fillna(0, inplace=True)

    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Handle date columns
    if 'Account_open_date' in df.columns:
        df['Account_open_date'] = pd.to_datetime(df['Account_open_date'], errors='coerce')
        df['Account_Age_Days'] = (pd.Timestamp.now() - df['Account_open_date']).dt.days
        df.drop(columns=['Account_open_date'], inplace=True)

    return df

# Split Data
def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train Gradient Boosting Classifier
def train_model(X_train, y_train, random_state=42):
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_pred_prob)
    print("AUC-ROC Score:", auc_roc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return y_pred, y_pred_prob

# Global SHAP Explanation
def global_shap_analysis(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    # Summary Plot
    print("\nSHAP Summary Plot:")
    shap.summary_plot(shap_values_to_plot, X_test, feature_names=X_test.columns)

    # Feature Importance Plot
    shap.summary_plot(shap_values_to_plot, X_test, plot_type="bar")
    
    return explainer, shap_values

# Local SHAP Explanation
def local_shap_analysis(explainer, shap_values, X_test, instance_idx):
    print(f"\nSHAP Force Plot for Instance {instance_idx}:")

    if isinstance(shap_values, list):
        shap_values_class = shap_values[1]
        expected_value_class = explainer.expected_value[1]
    else:
        shap_values_class = shap_values
        expected_value_class = explainer.expected_value

    shap.force_plot(
        expected_value_class,
        shap_values_class[instance_idx],
        X_test.iloc[instance_idx],
        matplotlib=True
    )
    
    # Save Force Plot as HTML
    shap.save_html("plots/Model Training/force_plot.html", shap.force_plot(
        expected_value_class,
        shap_values_class[instance_idx],
        X_test.iloc[instance_idx]
    ))
    
    # SHAP Dependence Plot
    print(f"\nSHAP Dependence Plot for Top Feature of Instance {instance_idx}:")
    top_feature_index = np.argmax(np.abs(shap_values_class[instance_idx]))
    shap.dependence_plot(top_feature_index, shap_values_class, X_test)

    # Waterfall Plot
    print(f"\nSHAP Waterfall Plot for Instance {instance_idx}:")
    shap.plots.waterfall(shap.Explanation(
        values=shap_values_class[instance_idx],
        base_values=expected_value_class,
        data=X_test.iloc[instance_idx],
        feature_names=X_test.columns
    ))

# Save Model and Results
def save_results(model, X_test, y_test, y_pred, y_pred_prob, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, 'credit_risk_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved as '{model_path}'.")

    # Save predictions
    X_test_copy = X_test.copy()
    X_test_copy['Actual'] = y_test
    X_test_copy['Predicted'] = y_pred
    X_test_copy['Probability'] = y_pred_prob
    X_test_copy.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print(f"Predictions saved to '{output_dir}/predictions.csv'.")
    
    X_test_copy['Risk Score'] = y_pred_prob * 1000
    X_test_copy.to_csv(os.path.join(output_dir, 'predictions_with_explanations.csv'), index=False)
    print(f"Predictions saved to '{output_dir}/predictions_with_explanations.csv'.")

    return model_path

if __name__ == "__main__":
    file_path = 'datasets/feature_extraction.csv'
    target_column = 'charge_off_status'
    output_dir = 'model_output'

    # Preprocess and split data
    df = preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)

    # Global SHAP Analysis
    explainer, shap_values = global_shap_analysis(model, X_test)

    # Local SHAP Analysis 
    instance_idx = 0
    local_shap_analysis(explainer, shap_values, X_test, instance_idx)

    # Save results
    save_results(model, X_test, y_test, y_pred, y_pred_prob, output_dir)