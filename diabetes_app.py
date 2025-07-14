import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load('diabetes_model.pkl')

st.title("🩺 Diabetes Risk Predictor")
st.write("Enter your health information below:")

age = st.slider("Age (in years)", 20, 80, 40)
bmi = st.number_input("BMI (Body Mass Index)", 15.0, 45.0, 25.0)
bp = st.number_input("Average Blood Pressure", 50.0, 120.0, 80.0)
s1 = st.slider("Total Cholesterol", 100, 300, 180)
s2 = st.slider("LDL Cholesterol", 100, 200, 130)
s3 = st.slider("HDL Cholesterol", 20, 100, 50)
s4 = st.slider("TCH/HDL ratio", 1, 10, 4)
s5 = st.number_input("Log of serum triglycerides", 4.0, 6.0, 5.0)
s6 = st.slider("Blood sugar level", 80, 200, 120)


sex = st.radio("Sex", ["Male", "Female"])
sex_value = 0.05 if sex == "Male" else -0.05  # Approximate encoding from sklearn dataset

input_data = pd.DataFrame([[age, sex_value, bmi, bp, s1, s2, s3, s4, s5, s6]],
                          columns=['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ High risk of diabetes! (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Low risk of diabetes. (Probability: {proba:.2f})")
    
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    feature_names = input_data.columns
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax, palette="viridis")
    st.pyplot(fig)
