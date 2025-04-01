import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from predict import predict_from_cv2, emotion_labels
import base64

def main():
    st.set_page_config(layout="wide")  # Set landscape mode

    # Load and display GIF as full background
    file_ = open("background_final.png", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/gif;base64,{data_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        .stSuccess {{
            width: 100%;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display logo next to the title
    col1, col2 = st.columns([0.15, 0.85])  # Adjust width ratio
    with col1:
        st.image("logo.png", width=170)  # Increase logo size
    with col2:
        st.title("Facial Expression Recognition")

    st.write("Upload an image to predict the facial expression.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        image = np.array(image)

        col1, col2 = st.columns([0.6, 0.4])  # Adjust column widths to bring button closer
        with col1:
            st.image(image, caption="Uploaded Image", width=250)  # Set a smaller width
        with col2:
            st.write("\n")  # Add space to align button closer
            if st.button("Predict Emotion"):
                model_path = "emotion_model.pth"
                class_id = predict_from_cv2(model_path, image)
                emotion = emotion_labels.get(class_id, "Unknown")
                st.success(f"Predicted Emotion: {emotion}")

if __name__ == "__main__":
    main()
