import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import numpy as np
import streamlit as st

# --- Load model safely with Streamlit caching to prevent retracing ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5', compile=False)

model = load_model()

# --- Load encoders and scaler ---
@st.cache_resource
def load_artifacts():
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return onehot_encoder_geo, label_encoder_gender, scaler

onehot_encoder_geo, label_encoder_gender, scaler = load_artifacts()

# --- Streamlit UI ---
st.title("💡 Customer Churn Prediction")

geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
age = st.slider('🎂 Age', 18, 92, 35)
balance = st.number_input('💰 Balance', min_value=0.0, value=0.0, step=100.0)
credit_score = st.number_input('📊 Credit Score', min_value=300, max_value=850, value=600)
estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)
tenure = st.slider('🕓 Tenure (years)', 0, 10, 3)
num_of_products = st.slider('📦 Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

# --- Prepare the input data ---
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# --- One-hot encode Geography safely ---
geo_encoded = onehot_encoder_geo.transform(pd.DataFrame({'Geography': [geography]})).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# --- Merge encoded + numeric features ---
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# --- Scale input ---
input_data_scaled = scaler.transform(input_data)

# --- Predict ---
prediction = model.predict(input_data_scaled)
prediction_proba = float(prediction[0][0])

# --- Display results ---
st.subheader(f"🔮 Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.error("🚨 The customer is **likely to churn.**")
else:
    st.success("✅ The customer is **not likely to churn.**")
