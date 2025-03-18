import shap as sh

"""
#explainer = sh.Explainer(model.get_model(), X_train)
#shap_values = explainer(X_test)

def GetSummaryPLot(dataFrame, X_test, shap_values):
    return sh.summary_plot(shap_values, X_test, feature_names=dataFrame.drop(columns=['Biopsy']).columns)

def GetDependencePlot(dataFrame, X_test, shap_values):
    return sh.dependence_plot('age', shap_values, X_test, feature_names=dataFrame.drop(columns=['Biopsy']).columns)

def GetDependencePlot(dataFrame, X_test, shap_values):
    return sh.force_plot(shap_values[0].base_values, shap_values[0].values, X_test[0], feature_names=dataFrame.drop(columns=['Biopsy']).columns)
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
import argparse


def load_model_and_data(model_path, data_path):
    """
    Load a saved model from a .pkl file and the associated data for SHAP explanations

    Parameters:
    model_path (str): Path to the .pkl model file
    data_path (str): Path to CSV data file

    Returns:
    tuple: (model, X_test, feature_names)
    """
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)

    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]
    feature_names = X.columns.tolist()

    # If dataset is large, use a sample
    if X.shape[0] > 10000:
        X_sample = X.sample(n=1000, random_state=42)
    else:
        X_sample = X

    return model, X_sample, feature_names


def create_shap_explainer(model, X_sample, model_type=None):
    """
    Create appropriate SHAP explainer based on model type

    Parameters:
    model: The machine learning model
    X_sample: Sample data for creating explainer
    model_type (str, optional): Type of model to use specific explainer

    Returns:
    shap.Explainer: SHAP explainer object
    """
    print("Creating SHAP explainer...")

    # Auto-detect model type if not specified
    if model_type is None:
        model_name = type(model).__name__.lower()

        if any(x in model_name for x in ['xgb', 'lgbm', 'catboost', 'forest', 'tree', 'gbm', 'boost']):
            print("Detected tree-based model, using TreeExplainer")
            return shap.TreeExplainer(model)
        elif any(x in model_name for x in ['svm', 'sgd', 'linear', 'logistic', 'regression']):
            print("Detected linear model, using LinearExplainer")
            return shap.LinearExplainer(model, X_sample)
        else:
            print("Model type not detected, using KernelExplainer")
            return shap.KernelExplainer(model.predict if hasattr(model, 'predict') else model, X_sample)

    # Use specified model type
    if model_type.lower() in ['xgboost', 'lightgbm', 'catboost', 'randomforest', 'tree']:
        return shap.TreeExplainer(model)
    elif model_type.lower() in ['linear', 'logistic', 'regression']:
        return shap.LinearExplainer(model, X_sample)
    else:
        predict_fn = model.predict if hasattr(model, 'predict') else model
        return shap.KernelExplainer(predict_fn, X_sample)


def generate_shap_values(explainer, X_sample):
    """
    Generate SHAP values using the explainer

    Parameters:
    explainer: SHAP explainer object
    X_sample: Sample data for explanation

    Returns:
    shap_values: Computed SHAP values
    """
    print("Calculating SHAP values (this may take some time)...")
    return explainer.shap_values(X_sample)


def create_output_directory(output_dir):
    """
    Create output directory if it doesn't exist

    Parameters:
    output_dir (str): Directory path
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")


def generate_shap_visualizations(shap_values, X_sample, feature_names, output_dir):
    """
    Generate and save various SHAP visualizations

    Parameters:
    shap_values: Computed SHAP values
    X_sample: Sample data used for explanations
    feature_names: List of feature names
    output_dir (str): Directory to save plots
    """
    print("Generating visualizations...")

    # For classification models with multiple classes
    if isinstance(shap_values, list):
        class_index = 1  # Usually 1 is the positive class (adjust if needed)
        values_to_plot = shap_values[class_index]
        title_suffix = f" (Class {class_index})"
    else:
        values_to_plot = shap_values
        title_suffix = ""

    # 1. Summary Plot (Feature Importance)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(values_to_plot, X_sample, feature_names=feature_names, show=False)
    plt.title(f"SHAP Feature Importance{title_suffix}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Bar Plot (Mean absolute SHAP value for each feature)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(values_to_plot, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance (Bar Plot){title_suffix}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_bar_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Dependence plots for top 5 features
    mean_abs_shap = np.mean(np.abs(values_to_plot), axis=0)
    feature_importance = pd.DataFrame(list(zip(feature_names, mean_abs_shap)),
                                      columns=['feature_name', 'feature_importance'])
    feature_importance.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Get top 5 features
    top_features = feature_importance.head(5)['feature_name'].tolist()

    for feature in top_features:
        try:
            plt.figure(figsize=(12, 8))
            feature_idx = feature_names.index(feature)
            shap.dependence_plot(feature_idx, values_to_plot, X_sample, feature_names=feature_names, show=False)
            plt.title(f"SHAP Dependence Plot for {feature}{title_suffix}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_dependence_{feature.replace(' ', '_')}.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating dependence plot for feature {feature}: {e}")

    # 4. Force plot for sample examples (first 3 examples)
    try:
        num_examples = min(3, X_sample.shape[0])
        for i in range(num_examples):
            plt.figure(figsize=(20, 3))
            force_plot = shap.force_plot(
                explainer.expected_value[class_index] if isinstance(explainer.expected_value, list)
                else explainer.expected_value,
                values_to_plot[i, :], X_sample.iloc[i, :],
                feature_names=feature_names, matplotlib=True, show=False)
            plt.title(f"SHAP Force Plot for Example {i + 1}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_force_plot_example_{i + 1}.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Error creating force plots: {e}")

    # 5. Decision plot for top 10 examples
    try:
        plt.figure(figsize=(12, 10))
        num_examples = min(10, X_sample.shape[0])
        shap.decision_plot(explainer.expected_value[class_index] if isinstance(explainer.expected_value, list)
                           else explainer.expected_value,
                           values_to_plot[:num_examples], X_sample.iloc[:num_examples],
                           feature_names=feature_names, show=False)
        plt.title(f"SHAP Decision Plot{title_suffix}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_decision_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating decision plot: {e}")

    print(f"All visualizations saved to {output_dir}")


parser = argparse.ArgumentParser(description='Generate SHAP explanation visualizations for a trained model.')
parser.add_argument('--model', type=str, required=True, help='Path to .pkl model file')
parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
parser.add_argument('--output', type=str, default='shap_results', help='Output directory for visualizations')
parser.add_argument('--model-type', type=str, help='Model type (tree, linear, kernel) for specific explainer')

args = parser.parse_args()

# Create output directory
create_output_directory(args.output)

# Load model and data
model, X_sample, feature_names = load_model_and_data(args.model, args.data)

# Create SHAP explainer
explainer = create_shap_explainer(model, X_sample, args.model_type)

# Generate SHAP values
shap_values = generate_shap_values(explainer, X_sample)

# Generate and save visualizations
generate_shap_visualizations(shap_values, X_sample, feature_names, args.output)

# Also save feature importance data to CSV
if isinstance(shap_values, list):
    class_index = 1  # Usually 1 is the positive class
    values_to_analyze = shap_values[class_index]
else:
    values_to_analyze = shap_values

# Calculate mean absolute SHAP values for each feature
mean_abs_shap = np.mean(np.abs(values_to_analyze), axis=0)

# Create DataFrame with feature importance information
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Absolute_SHAP_Value': mean_abs_shap,
    'Relative_Importance': mean_abs_shap / np.sum(mean_abs_shap) * 100
})

# Sort by importance
feature_importance = feature_importance.sort_values('Mean_Absolute_SHAP_Value', ascending=False)

# Save to CSV
importance_file = os.path.join(args.output, 'feature_importance.csv')
feature_importance.to_csv(importance_file, index=False)
print(f"Feature importance saved to {importance_file}")

print("SHAP analysis completed successfully!")


