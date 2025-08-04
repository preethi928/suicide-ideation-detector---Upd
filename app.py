import streamlit as st
import re
import joblib

# Load saved files
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_text(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    return "Suicidal" if pred == 1 else "Non-Suicidal"

st.title("Suicide Ideation Detection")
user_input = st.text_area("Enter text to analyze:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.write("Please enter some text.")
    else:
        result = predict_text(user_input)
        st.write(f"Prediction: **{result}**")
