"""
import streamlit as st
from ProAndTrain import model

# Streamlit Interface
st.title("Cervical Cancer Risk Prediction with SHAP Explanations")

# Model Selection
model_choice = st.selectbox("Select Model", options=['Random Forest', 'SVM', 'XGBoost', 'CatBoost'])

# Initialize the selected model
model = model.MLModel(model_choice)
model.train(X_train, y_train)

# Feature Inputs (20 float inputs for the features)
input_features = [st.number_input(f'Feature {i + 1}', min_value=-100.0, max_value=100.0, value=0.0, step=0.1) for i in range(20)]
input_features = np.array(input_features).reshape(1, -1)  # Reshape to make it a single row input

# Predict the result (output: bool)
prediction = model.predict(input_features)
st.write(f"Prediction (Biopsy): {'Positive' if prediction[0] == 1 else 'Negative'}")

# SHAP Explanations
explainer = shap.Explainer(model.get_model(), X_train)
shap_values = explainer(input_features)

# Plot SHAP summary (showing feature importance)
st.subheader("SHAP Summary Plot")
shap.summary_plot(shap_values, input_features, feature_names=[f"Feature {i + 1}" for i in range(20)])

# Show SHAP force plot for a specific instance
st.subheader("SHAP Force Plot")
shap.force_plot(shap_values[0].base_values, shap_values[0].values, input_features[0], feature_names=[f"Feature {i + 1}" for i in range(20)])"
"""

import sklearn
import streamlit as st

import pandas as pd
import numpy as np


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from ProAndTrain import dataProcessing
from ProAndTrain import model

import shap


df = pd.read_csv("data/risk_factors_cervical_cancer.csv")

df = dataProcessing.process_data(df)

X = np.array(df.drop(columns = ['Biopsy'])).astype('float32')
y = np.array(df['Biopsy']).astype('float32')
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_test, x_val, y_test, y_val = train_test_split(X, y, test_size = 0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

"""
# Show the first few rows of the dataframe
st.write("### Data Preview")
st.write(df.head())

# Display DataFrame info
st.write("### DataFrame Info")
st.write(df.info())

# Slider for selecting the number of rows to display
num_rows = st.slider("Select number of rows to display", min_value=5, max_value=df.shape[0], step=5)

# Display the selected number of rows
st.write(f"### Displaying the first {num_rows} rows of the DataFrame:")
st.write(df.head(num_rows))

# Optional: Display the entire dataframe (if not too large)
if st.checkbox("Show full DataFrame"):
    st.write(df)"
"""
st.title("Cervical Cancer Risk Prediction with SHAP Explanations")

# Model Selection
model_choice = st.selectbox("Select Model", options=['Random Forest', 'SVM', 'XGBoost', 'CatBoost'])

# Initialize the selected model
Model = model.MLModel(model_choice)
Model.train(X_train, y_train)

# Feature Inputs (20 float inputs for the features)
input_features = [st.number_input(f'Feature {i + 1}', min_value=-100.0, max_value=100.0, value=0.0, step=0.1) for i in range(20)]
input_features = np.array(input_features).reshape(1, -1)  # Reshape to make it a single row input

# Predict the result (output: bool)
prediction = model.predict(input_features)
st.write(f"Prediction (Biopsy): {'Positive' if prediction[0] == 1 else 'Negative'}")

# SHAP Explanations
explainer = shap.Explainer(model.get_model(), X_train)
shap_values = explainer(input_features)

# Plot SHAP summary (showing feature importance)
st.subheader("SHAP Summary Plot")
shap.summary_plot(shap_values, input_features, feature_names=[f"Feature {i + 1}" for i in range(20)])

# Show SHAP force plot for a specific instance
st.subheader("SHAP Force Plot")
shap.force_plot(shap_values[0].base_values, shap_values[0].values, input_features[0], feature_names=[f"Feature {i + 1}" for i in range(20)])
