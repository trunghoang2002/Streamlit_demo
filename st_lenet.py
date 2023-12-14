import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Load LeNet model
model = load_model("lenet_mnist_model.h5")
# Define class labels for MNIST
class_labels = [str(i) for i in range(10)]

st.title("MNIST Digit Classification with LeNet")

uploaded_file = st.file_uploader("Upload an image to predict...", type="jpg")

if uploaded_file is not None:
    # files = {"file": uploaded_file}
    content = uploaded_file.read()
    image = Image.open(io.BytesIO(content)).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))
    image_array = np.array(image).reshape((1, 28, 28, 1)).astype("float32") / 255

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    st.write(f"Predicted Digit: {predicted_label}")
    
    # response = requests.post("http://127.0.0.1:8000/predict", files=files)
    # result = response.json()
    # st.write(f"Predicted Digit: {result['predicted_label']}")
