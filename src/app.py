import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Set up the Streamlit page
st.set_page_config(page_title="Skincare Recognition", page_icon="🔬")

# Load the model and processor
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
    model = AutoModelForImageClassification.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
    return processor, model

processor, model = load_model()

st.title("Skincare Recognition Model")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # Get the predicted class
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    # Display the prediction
    st.subheader("Prediction:")
    st.write(f"The model predicts this image is: **{predicted_class}**")

    # Display confidence scores
    st.subheader("Confidence Scores:")
    probs = torch.nn.functional.softmax(logits, dim=1)
    for i, p in enumerate(probs[0]):
        st.write(f"{model.config.id2label[i]}: {p.item():.2%}")

# Add some information about the model
st.sidebar.header("About")
st.sidebar.write("""
This application uses a pre-trained model from Hugging Face for skin cancer image classification.
The model can identify various types of skin conditions based on the uploaded image.

Please note that this is not a substitute for professional medical advice. Always consult with a healthcare professional for proper diagnosis and treatment.
""")

