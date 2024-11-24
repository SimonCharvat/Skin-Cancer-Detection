import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Set page config
st.set_page_config(page_title="Skin Cancer Recognition", page_icon="🔬", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #111827;
        color: #9CA3AF;
    }
    
    .stButton>button {
        color: white;
        background-color: #10B981;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #059669;
    }
    .uploadedFile {
        color: #9CA3AF;
    }
    .css-1kyxreq {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and processor
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
    model = AutoModelForImageClassification.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
    return processor, model

processor, model = load_model()

# Custom HTML layout
st.markdown("""
<section class="text-gray-400 bg-gray-900 body-font">
  <div class="container px-5 py-24 mx-auto flex flex-wrap items-center">
    <div class="lg:w-3/5 md:w-1/2 md:pr-16 lg:pr-0 pr-0">
      <h1 class="title-font font-medium text-3xl text-white">Skin Cancer Recognition Model</h1>
      <p class="leading-relaxed mt-4">Upload an image of a skin lesion to get a prediction on the type of skin condition. This tool uses advanced AI to assist in early detection and classification of skin cancer.</p>
    </div>
    <div class="lg:w-2/6 md:w-1/2 bg-gray-800 bg-opacity-50 rounded-lg p-8 flex flex-col md:ml-auto w-full mt-10 md:mt-0">
      <h2 class="text-white text-lg font-medium title-font mb-5">Upload Image</h2>
      <div id="file_uploader"></div>
      <div id="image_display"></div>
      <div id="prediction_results"></div>
    </div>
  </div>
</section>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Make a prediction
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # Get the predicted class
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    # Display the prediction
    st.markdown("<h3 class='text-white text-lg font-medium title-font mb-3'>Prediction:</h3>", unsafe_allow_html=True)
    st.markdown(f"<p>The model predicts this image is: <strong>{predicted_class}</strong></p>", unsafe_allow_html=True)

    # Display confidence scores
    st.markdown("<h3 class='text-white text-lg font-medium title-font mb-3 mt-4'>Confidence Scores:</h3>", unsafe_allow_html=True)
    probs = torch.nn.functional.softmax(logits, dim=1)
    for i, p in enumerate(probs[0]):
        st.markdown(f"<p>{model.config.id2label[i]}: {p.item():.2%}</p>", unsafe_allow_html=True)

# Add information about the model in the sidebar