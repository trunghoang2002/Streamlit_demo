import streamlit as st
import requests

# st.sidebar.title("side bar")
# st.sidebar.header("header")
# st.sidebar.button("Click")

st.title("MNIST Digit Classification with LeNet")

uploaded_file = st.file_uploader("Upload an image to predict...", type="jpg")

if uploaded_file is not None:
    files = {"file": uploaded_file}
    response = requests.post("http://127.0.0.1:8000/predict", files=files)
    result = response.json()
    st.write(f"Predicted Digit: {result['predicted_label']}")