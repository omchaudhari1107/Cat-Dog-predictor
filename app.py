import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import joblib
import pywt
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('model_cat_dog.pkl')

def w2d(img, mode='haar', level=1):
    imArray = np.array(img)
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H

def preprocess_image(image):
    image = ImageOps.fit(image, (200, 200), Image.LANCZOS)
    img_array = np.array(image)
    img_har = w2d(img_array, 'db1', 5)
    combined_img = np.vstack((img_array.reshape(200 * 200 * 3, 1), img_har.reshape(200 * 200, 1)))
    combined_img = combined_img.reshape(1, -1).astype(float)
    return combined_img

# Streamlit app
st.title("Dog and Cat Image Recognition")
st.write("Upload an image of a dog or a cat, and the app will recognize it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        processed_image = preprocess_image(image)
        prediction_probabilities = model.predict_proba(processed_image)[0]
        labels = ['Cat', 'Dog']
        probabilities = {label: prob for label, prob in zip(labels, prediction_probabilities)}
        
        # Use Matplotlib to create a bar chart
        fig, ax = plt.subplots()
        ax.bar(probabilities.keys(), probabilities.values(), color=['blue', 'orange'])
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        st.pyplot(fig)
    except Exception as e:
        st.write("Error processing the image. Please try again.")
        st.write(f"Error details: {e}")
else:
    st.write("Please upload an image file.")
