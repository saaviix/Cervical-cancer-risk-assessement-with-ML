import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import os
import glob

st.set_page_config(page_title="Cervical Cancer Risk Predictor", layout="wide")

st.title("Cervical Cancer Risk Prediction with SHAP Explanations")
st.markdown("Select one of your existing model files to get risk predictions with SHAP explanations.")

cervical_cancer_features = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD",
    "IUD (years)",
    "STDs",
    "STDs (number)",
    "STDs: Number of diagnosis",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "DxHinselmann",
    "Schiller",
    "Citology"
]

feature_descriptions = {
    "Age": "Patient's age in years",
    "Number of sexual partners": "Number of sexual partners the patient has had",
    "First sexual intercourse": "Age at first sexual intercourse",
    "Num of pregnancies": "Number of pregnancies the patient has had",
    "Smokes": "Whether the patient smokes (0=No, 1=Yes)",
    "Smokes (years)": "Number of years the patient has smoked",
    "Smokes (packs/year)": "Number of packs per year the patient smokes",
    "Hormonal Contraceptives": "Whether the patient uses hormonal contraceptives (0=No, 1=Yes)",
    "Hormonal Contraceptives (years)": "Number of years on hormonal contraceptives",
    "IUD": "Whether the patient uses an intrauterine device (0=No, 1=Yes)",
    "IUD (years)": "Number of years using an intrauterine device",
    "STDs": "Whether the patient has had any STDs (0=No, 1=Yes)",
    "STDs (number)": "Number of sexually transmitted diseases the patient has had",
    "STDs: Number of diagnosis": "Number of STD diagnoses",
    "Dx:Cancer": "Previous diagnosis of cancer (0=No, 1=Yes)",
    "Dx:CIN": "Previous diagnosis of cervical intraepithelial neoplasia (0=No, 1=Yes)",
    "Dx:HPV": "Previous diagnosis of HPV (0=No, 1=Yes)",
    "DxHinselmann": "Hinselmann test result (0=Negative, 1=Positive)",
    "Schiller": "Schiller test result (0=Negative, 1=Positive)",
    "Citology": "Cytology test result (0=Negative, 1=Positive)"
}

pkl_files = glob.glob("*.pkl")

st.subheader("Select Model")

if not pkl_files:
    st.warning("No .pkl files found in the current directory.")
    selected_model_file = None
else:
    st.write("Select one of your existing model files:")
    
    model_cols = st.columns(min(4, len(pkl_files)))
    selected_model_file = None
    
    for i, file in enumerate(pkl_files):
        col_idx = i % len(model_cols)
        with model_cols[col_idx]:
            if st.button(f"{file}", key=f"model_{i}"):
                selected_model_file = file
                st.success(f"Selected model: {file}")

if selected_model_file is not None:
    st.subheader("Patient Features")
    st.markdown("Enter values for the patient's features:")
    
    col1, col2 = st.columns(2)
    
    features = []
    feature_values = []
    
    for i, feature in enumerate(cervical_cancer_features):
        current_col = col1 if i < 10 else col2
        with current_col:
            with st.expander(f"{feature} ℹ️"):
                st.write(feature_descriptions[feature])
                if feature in ["Smokes", "Hormonal Contraceptives", "IUD", "STDs", 
                              "Dx:Cancer", "Dx:CIN", "Dx:HPV", "DxHinselmann", "Schiller", "Citology"]:
                    feature_value = st.selectbox(f"Value", [0, 1], key=f"value_{i}")
                else:
                    feature_value = st.number_input(f"Value", min_value=0.0, value=0.0, format="%.2f", key=f"value_{i}")
            features.append(feature)
            feature_values.append(feature_value)
    
    predict_button = st.button("Generate Risk Prediction and Explanation")
    
    if predict_button:
        try:
            with st.spinner("Loading model and generating cervical cancer risk explanation..."):
                with open(selected_model_file, 'rb') as f:
                    model = pickle.load(f)
                st.info(f"Using model: {selected_model_file}")
                
                input_df = pd.DataFrame([feature_values], columns=features)
                
                if hasattr(model, 'predict_proba'):
                    try:
                        prediction_proba = model.predict_proba(input_df)
                        prediction = model.predict(input_df)
                        classification = True
                    except:
                        prediction = model.predict(input_df)
                        classification = False
                else:
                    prediction = model.predict(input_df)
                    classification = False
                
                st.subheader("Risk Prediction Results")
                if classification:
                    pred_class = prediction[0]
                    class_label = "High Risk" if pred_class == 1 else "Low Risk"
                    
                    st.markdown(f"### Predicted Risk: {class_label}")
                    
                    risk_proba = prediction_proba[0][1] if len(prediction_proba[0]) > 1 else 0.5
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.set_aspect('equal')
                        ax.axis('off')
                        
                        theta = np.linspace(0, np.pi, 100)
                        x = 0.5 + 0.4 * np.cos(theta)
                        y = 0.5 + 0.4 * np.sin(theta)
                        ax.plot(x, y, color='black', linewidth=2)
                        
                        for i in range(100):
                            angle = i * np.pi / 100
                            radius = 0.4
                            x_pos = 0.5 + radius * np.cos(angle)
                            y_pos = 0.5 + radius * np.sin(angle)
                            ax.plot([0.5, x_pos], [0.5, y_pos], color=plt.cm.RdYlGn_r(i/100), alpha=0.7, linewidth=3)
                        
                        needle_angle = np.pi * risk_proba
                        x_needle = 0.5 + 0.4 * np.cos(needle_angle)
                        y_needle = 0.5 + 0.4 * np.sin(needle_angle)
                        ax.plot([0.5, x_needle], [0.5, y_needle], color='black', linewidth=3)
                        
                        ax.text(0.1, 0.2, 'Low Risk', fontsize=12, ha='center')
                        ax.text(0.9, 0.2, 'High Risk', fontsize=12, ha='center')
                        ax.text(0.5, 0.1, f'Risk Score: {risk_proba:.2f}', fontsize=14, ha='center', fontweight='bold')
                        
                        st.pyplot(fig)
                    
                    st.write("Risk Probability Breakdown:")
                    proba_df = pd.DataFrame({
                        "Risk Category": ["Low Risk", "High Risk"],
                        "Probability": [prediction_proba[0][0], prediction_proba[0][1]] if len(prediction_proba[0]) > 1 else [1-risk_proba, risk_proba]
                    })
                    st.dataframe(proba_df)
                else:
                    st.metric("Risk Score", round(prediction[0], 4))
                
                st.subheader("Feature Importance for Risk Prediction")
                st.markdown("This shows how each feature contributes to the risk prediction:")
                
                if isinstance(model, Pipeline):
                    model_to_explain = model.steps[-1][1]
                    processed_input = model[:-1].transform(input_df)
                else:
                    model_to_explain = model
                    processed_input = input_df
                
                try:
                    explainer = shap.Explainer(model_to_explain)
                    shap_values = explainer(processed_input)
                    
                    st.write("### Individual Feature Contributions:")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if classification and hasattr(model, 'predict_proba'):
                        if len(shap_values.shape) > 2:
                            class_idx = 1
                            shap.plots.waterfall(shap_values[0, :, class_idx], max_display=20, show=False)
                        else:
                            shap.plots.waterfall(shap_values[0], max_display=20, show=False)
                    else:
                        shap.plots.waterfall(shap_values[0], max_display=20, show=False)
                    
                    st.pyplot(fig)
                    
                    st.write("### Most Important Risk Factors:")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if classification and hasattr(model, 'predict_proba'):
                        if len(shap_values.shape) > 2:
                            class_idx = 1
                            shap.summary_plot(shap_values[:, :, class_idx], processed_input, plot_type="bar", show=False)
                        else:
                            shap.summary_plot(shap_values, processed_input, plot_type="bar", show=False)
                    else:
                        shap.summary_plot(shap_values, processed_input, plot_type="bar", show=False)
                    
                    st.pyplot(fig)
                    
                    st.write("### Risk Factor Analysis:")
                    st.markdown("Red points indicate higher feature values, blue points indicate lower values. Position to the right means higher risk contribution.")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if classification and hasattr(model, 'predict_proba'):
                        if len(shap_values.shape) > 2:
                            class_idx = 1
                            shap.summary_plot(shap_values[:, :, class_idx], processed_input, show=False)
                        else:
                            shap.summary_plot(shap_values, processed_input, show=False)
                    else:
                        shap.summary_plot(shap_values, processed_input, show=False)
                    
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error generating SHAP values: {str(e)}")
                    st.write("SHAP explanation failed. This might be due to model incompatibility with SHAP.")
                    
                    if hasattr(model_to_explain, 'feature_importances_'):
                        st.write("Feature Importance (from model):")
                        importance_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': model_to_explain.feature_importances_
                        }).sort_values(by='Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.barh(importance_df['Feature'], importance_df['Importance'])
                        plt.xlabel('Importance')
                        plt.title('Feature Importance for Cervical Cancer Risk')
                        st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Error loading model or generating prediction: {str(e)}")
            st.write("Make sure your selected model file is a valid pickled model file.")

with st.expander("About Cervical Cancer Risk Factors"):
    st.markdown("""
    ### Key Risk Factors for Cervical Cancer
    
    Cervical cancer is primarily caused by human papillomavirus (HPV) infection. However, several other factors can influence the risk:
    
    **Demographic Factors:**
    - Age (risk increases in mid-life)
    - Early sexual activity
    - Multiple sexual partners
    
    **Lifestyle Factors:**
    - Smoking
    - Long-term use of hormonal contraceptives
    
    **Medical History:**
    - STD history
    - Multiple pregnancies
    - Previous HPV diagnosis
    - Previous CIN (Cervical Intraepithelial Neoplasia) diagnosis
    
    **Diagnostic Tests:**
    - Hinselmann test
    - Schiller test
    - Cytology test
    """)