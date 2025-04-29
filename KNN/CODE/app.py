import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define custom CSS for styling
def load_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header-container {
        background-color: #ff4b4b;
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-container {
        background-color: #f1f3f5;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .feature-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #6c757d;
    }
    .result-positive {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .result-negative {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_model():
    try:
        # Adjust the path if needed
        model_path = 'C:\Course\Heart Disease UCI Dataset\CODE\KNN\heart_disease_knn_model.pkl'
        # Try absolute path if relative path fails
        if not os.path.exists(model_path):
            model_path = r"C:\Course\Heart Disease UCI Dataset\CODE\KNN\heart_disease_knn_model.pkl"
        
        model = joblib.load(model_path)
        # Get feature names if available
        feature_names = None
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        elif hasattr(model, 'steps') and hasattr(model.steps[0][1], 'feature_names_in_'):
            # For pipeline models, check the first transformer
            feature_names = model.steps[0][1].feature_names_in_
        
        return model, feature_names
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'heart_disease_knn_model.pkl' is in the current directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to inspect the model pipeline
def inspect_model(model):
    if model is None:
        return "Model not loaded"
    
    info = []
    
    # Check if it's a pipeline
    if hasattr(model, 'steps'):
        info.append(f"Model is a Pipeline with {len(model.steps)} steps:")
        for i, (name, step) in enumerate(model.steps):
            info.append(f"  Step {i+1}: {name} - {type(step).__name__}")
            
            # Check for specific transformers that might have feature info
            if hasattr(step, 'n_features_in_'):
                info.append(f"    Expected features: {step.n_features_in_}")
            if hasattr(step, 'feature_names_in_'):
                info.append(f"    Feature names: {step.feature_names_in_}")
    else:
        info.append(f"Model type: {type(model).__name__}")
        
    if hasattr(model, 'n_features_in_'):
        info.append(f"Expected features: {model.n_features_in_}")
        
    return "\n".join(info)

# Function to make prediction
def predict_heart_disease(model, features, feature_names=None):
    try:
        features_array = np.array(features).reshape(1, -1)
        
        # If we have feature names, check if we need to adjust
        if feature_names is not None:
            # Create a DataFrame with the right feature names
            # This handles missing features by setting them to default values
            feature_dict = {}
            
            # Use the features we have
            for i, name in enumerate(feature_names):
                if i < len(features):
                    feature_dict[name] = features[i]
                else:
                    # For any missing features, use a default value (0)
                    feature_dict[name] = 0
                    
            features_df = pd.DataFrame([feature_dict])
            # Convert back to numpy array in the right order
            features_array = features_df.values
            
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array)
        return prediction[0], probability[0][1]  # Return prediction and probability of heart disease
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Function to create feature tooltips
def get_tooltip(feature):
    tooltips = {
        'age': "Age in years",
        'sex': "Gender (1 = male, 0 = female)",
        'cp': "Chest pain type (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic)",
        'trestbps': "Resting blood pressure in mm Hg on admission to the hospital",
        'chol': "Serum cholesterol in mg/dl",
        'fbs': "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
        'restecg': "Resting electrocardiographic results",
        'thalach': "Maximum heart rate achieved",
        'exang': "Exercise induced angina (1 = yes, 0 = no)",
        'oldpeak': "ST depression induced by exercise relative to rest",
        'slope': "Slope of the peak exercise ST segment",
        'ca': "Number of major vessels (0-3) colored by fluoroscopy",
        'thal': "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)"
    }
    return tooltips.get(feature, "")

# Main function
def main():
    load_css()
    
    # Load the model and get feature names
    model, feature_names = load_model()
    
    # Header
    st.markdown('<div class="header-container"><h1>❤️ Heart Disease Prediction</h1><p>Fill in the patient details to predict heart disease risk</p></div>', unsafe_allow_html=True)
    
    # Display model information in debug mode
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
        
    with st.sidebar:
        st.title("Debug Options")
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
        
        if st.session_state.debug_mode and model is not None:
            st.subheader("Model Information")
            st.text(inspect_model(model))
            
            if feature_names is not None:
                st.subheader("Expected Features")
                st.write(f"Count: {len(feature_names)}")
                st.write(list(feature_names))
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    # Form for input features
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("Patient Demographics")
        
        age = st.slider("Age", min_value=20, max_value=100, value=50, help=get_tooltip('age'))
        
        sex = st.radio("Gender", options=["Female", "Male"])
        sex_value = 1 if sex == "Male" else 0
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("Heart Examination")
        
        cp_options = {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }
        cp = st.selectbox(
            "Chest Pain Type", 
            options=list(cp_options.keys()),
            format_func=lambda x: cp_options[x],
            help=get_tooltip('cp')
        )
        
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120, help=get_tooltip('trestbps'))
        
        chol = st.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, help=get_tooltip('chol'))
        
        fbs_options = {0: "No (≤120 mg/dl)", 1: "Yes (>120 mg/dl)"}
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl", 
            options=list(fbs_options.keys()),
            format_func=lambda x: fbs_options[x],
            help=get_tooltip('fbs')
        )
        
        restecg_options = {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }
        restecg = st.selectbox(
            "Resting ECG Results", 
            options=list(restecg_options.keys()),
            format_func=lambda x: restecg_options[x],
            help=get_tooltip('restecg')
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("Exercise Test Results")
        
        thalach = st.slider("Maximum Heart Rate", min_value=70, max_value=220, value=150, help=get_tooltip('thalach'))
        
        exang_options = {0: "No", 1: "Yes"}
        exang = st.selectbox(
            "Exercise Induced Angina", 
            options=list(exang_options.keys()),
            format_func=lambda x: exang_options[x],
            help=get_tooltip('exang')
        )
        
        oldpeak = st.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1, help=get_tooltip('oldpeak'))
        
        slope_options = {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment", 
            options=list(slope_options.keys()),
            format_func=lambda x: slope_options[x],
            help=get_tooltip('slope')
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("Additional Tests")
        
        ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0, help=get_tooltip('ca'))
        
        # Fix for the error - thalassemia options
        thal_options = {
            1: "Normal",
            2: "Fixed Defect",
            3: "Reversible Defect"
        }
        thal = st.selectbox(
            "Thalassemia", 
            options=list(thal_options.keys()),
            format_func=lambda x: thal_options[x],
            help=get_tooltip('thal')
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create feature array
    features = [age, sex_value, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    # Show the current feature vector in debug mode
    if st.session_state.debug_mode:
        st.subheader("Current Feature Vector")
        st.write(features)
        st.write(f"Number of features: {len(features)}")
    
    # Prediction section
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Predict Heart Disease Risk", use_container_width=True):
            if model is not None:
                with st.spinner("Predicting..."):
                    prediction, probability = predict_heart_disease(model, features, feature_names)
                
                if prediction is not None and probability is not None:
                    if prediction == 1:
                        st.markdown(f'<div class="result-positive">Heart Disease Detected<br>Risk probability: {probability:.2%}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-negative">No Heart Disease Detected<br>Risk probability: {probability:.2%}</div>', unsafe_allow_html=True)
                    
                    # Display gauge chart for risk visualization
                    fig, ax = plt.subplots(figsize=(8, 2))
                    
                    # Create gauge chart
                    risk_categories = ['Low Risk', 'Moderate Risk', 'High Risk']
                    ax.barh([0], [0.33], color='green', alpha=0.8)
                    ax.barh([0], [0.33], left=0.33, color='yellow', alpha=0.8)
                    ax.barh([0], [0.34], left=0.66, color='red', alpha=0.8)
                    
                    # Add marker for the predicted probability
                    ax.scatter(probability, 0, color='black', s=300, zorder=5, marker='|')
                    
                    # Add labels
                    ax.set_yticks([])
                    ax.set_xticks([0, 0.33, 0.66, 1])
                    ax.set_xticklabels(['0%', '33%', '66%', '100%'])
                    ax.set_xlim(0, 1)
                    ax.set_title('Heart Disease Risk Level')
                    
                    # Remove spines
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    
                    st.pyplot(fig)
                    
                    # Feature importance visualization
                    st.subheader("Key Contributing Factors")
                    
                    # Map feature names to more readable labels
                    feature_names_readable = [
                        'Age', 'Gender', 'Chest Pain Type', 'Resting BP', 'Cholesterol',
                        'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 
                        'Exercise Angina', 'ST Depression', 'ST Slope', 
                        'Major Vessels', 'Thalassemia'
                    ]
                    
                    # This is just for visualization purposes
                    st.write("The following factors are generally important in heart disease prediction:")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info("**Age**: {:.0f} years".format(age))
                        st.info("**Chest Pain Type**: {}".format(cp_options[cp]))
                        st.info("**Cholesterol**: {} mg/dl".format(chol))
                    
                    with col2:
                        st.info("**Maximum Heart Rate**: {}".format(thalach))
                        st.info("**ST Depression**: {:.1f}".format(oldpeak))
                        st.info("**Major Vessels**: {}".format(ca))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Information section
    with st.expander("About Heart Disease Prediction"):
        st.write("""
        ## Understanding Heart Disease Risk Factors
        
        This application uses a K-Nearest Neighbors (KNN) machine learning model trained on the UCI Heart Disease dataset. 
        The model examines multiple risk factors to predict the likelihood of heart disease.
        
        ### Key Risk Factors:
        
        - **Age and Gender**: Risk increases with age, and men generally have a higher risk than women.
        - **Chest Pain**: Different types of chest pain indicate varying levels of risk.
        - **Blood Pressure and Cholesterol**: Higher levels often correlate with increased heart disease risk.
        - **Diabetes**: Indicated by fasting blood sugar levels.
        - **ECG Results**: Abnormalities can indicate heart problems.
        - **Maximum Heart Rate**: Lower maximum heart rates may indicate reduced cardiac function.
        - **Exercise-Induced Angina**: Chest pain during exercise is a significant indicator.
        
        ### Disclaimer:
        This tool is for educational purposes only and should not replace professional medical advice.
        """)
    
    # Footer
    st.markdown('<div class="footer">© 2025 Heart Disease Predictor - For educational purposes only</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()