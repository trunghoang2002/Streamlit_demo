# fastapi_app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load LeNet model
model = load_model("lenet_mnist_model.h5")

# Define class labels for MNIST
class_labels = [str(i) for i in range(10)]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))
    image_array = np.array(image).reshape((1, 28, 28, 1)).astype("float32") / 255

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]

    return JSONResponse(content={"predicted_label": predicted_label})