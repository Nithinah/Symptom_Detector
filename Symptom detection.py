import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import requests
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load dataset and model paths
DATA_PATH = "Disease_symptom_and_patient_profile_dataset.csv"
MODEL_PATH = "symptom_checker.pkl"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    label_encoders = {}
    feature_columns = ["Fever", "Cough", "Fatigue", "Difficulty Breathing",
                       "Gender", "Blood Pressure", "Cholesterol Level", "Age"]
    
    for col in feature_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

df, label_encoders = load_data()

# Train model only if not already saved
if not os.path.exists(MODEL_PATH):
    X = df.drop(columns=["Disease", "Outcome Variable"])
    y = df["Outcome Variable"]
    model = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=42)
    model.fit(X, y)
    pickle.dump(model, open(MODEL_PATH, "wb"))

# Load trained model
model = pickle.load(open(MODEL_PATH, "rb"))

# Streamlit App UI
st.title("🏥 AI-Powered Symptom Checker")
st.write("Enter your symptoms to get an AI-driven diagnosis and medical insights.")

# User Inputs
fever = st.radio("Do you have a fever?", ["Yes", "No"])
cough = st.radio("Do you have a cough?", ["Yes", "No"])
fatigue = st.radio("Do you feel fatigued?", ["Yes", "No"])
difficulty_breathing = st.radio("Do you have difficulty breathing?", ["Yes", "No"])
gender = st.selectbox("Select your gender", ["Male", "Female"])
bp = st.selectbox("Blood Pressure Level", ["Low", "Normal", "High"])
cholesterol = st.selectbox("Cholesterol Level", ["Low", "Normal", "High"])
age = st.slider("Select your age", 0, 100, 25)

# Convert user inputs to numerical values using LabelEncoder
try:
    input_data = np.array([
        label_encoders["Fever"].transform([fever])[0],
        label_encoders["Cough"].transform([cough])[0],
        label_encoders["Fatigue"].transform([fatigue])[0],
        label_encoders["Difficulty Breathing"].transform([difficulty_breathing])[0],
        label_encoders["Gender"].transform([gender])[0],
        label_encoders["Blood Pressure"].transform([bp])[0],
        label_encoders["Cholesterol Level"].transform([cholesterol])[0],
        age
    ]).reshape(1, -1)

    if input_data.shape[1] != model.n_features_in_:
        st.error(f"Feature mismatch! Expected {model.n_features_in_} features, but got {input_data.shape[1]}.")
        st.stop()

except KeyError as e:
    st.error(f"Error in input transformation: {e}. Please check input values.")
    st.stop()

# Hugging Face Inference API for AI Explanation
hf_token = st.secrets["general"].get("hf_token", None)  # Fetch token safely
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"

def get_ai_explanation(symptoms, diagnosis):
    if not hf_token:
        return "⚠️ AI explanation is unavailable. Please check the API token."

    headers = {"Authorization": f"Bearer {hf_token}"}
    prompt = f"Patient symptoms: {', '.join(symptoms)}. Diagnosis: {diagnosis}. Explain this diagnosis in simple terms."

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        response_json = response.json()

        if isinstance(response_json, list) and "generated_text" in response_json[0]:
            return response_json[0]["generated_text"]
        else:
            return "⚠️ AI model error: Invalid response from Hugging Face API."
    except Exception as e:
        return f"⚠️ AI model error: {str(e)}"

# Check Symptoms Button
if st.button("🔍 Check Symptoms"):
    prediction = model.predict(input_data)
    diagnosis = "Unknown Condition"

    if "Outcome Variable" in label_encoders:
        diagnosis = label_encoders["Outcome Variable"].inverse_transform(prediction)[0]

    symptoms = [fever, cough, fatigue, difficulty_breathing]
    ai_response = get_ai_explanation(symptoms, diagnosis)

    st.success(f"### 🏥 Diagnosis: {diagnosis}")
    st.write("#### 🤖 AI Explanation:")
    st.write(ai_response)
