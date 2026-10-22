import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_cnn_final.keras")

model = load_model()


class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]



st.title("🧠 Brain Tumor MRI Classifier")

st.write(
    "Upload a brain MRI image to classify it into one of four "
    "categories using a Convolutional Neural Network."
)



uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )

    # Preprocess
    img = image.resize((224, 224))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = predictions[0][predicted_index] * 100


    st.subheader("Prediction")

    st.success(
        f"Predicted Class: {predicted_class}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    st.subheader("Class Probabilities")

    probabilities = {
        class_names[i]: float(predictions[0][i])
        for i in range(len(class_names))
    }

    st.bar_chart(probabilities)