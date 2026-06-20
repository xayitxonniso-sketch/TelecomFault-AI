import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vector.pkl", "rb"))

st.set_page_config(page_title="TelecomFault AI", page_icon="📡")

st.title("📡 TelecomFault AI - Hackathon Project")
st.write("AI detects telecom network issues from user complaints")

text = st.text_input("Enter complaint:")

if text:
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    st.success("Predicted Issue:")
    st.markdown(f"### ⚠️ {prediction}")