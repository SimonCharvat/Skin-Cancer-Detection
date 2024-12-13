import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

import requests
from io import BytesIO
from typing import Literal



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

def set_color(key: str, style: Literal["red", "green", "yellow", "grey"], align: Literal["left", "center", "right"]="left") -> None:
  """
  Allows to dynamicaly change CSS style of Streamlit elements. The element must have defined key.
  When this function is called, it creates/overwrites CSS style made only for the element.
  This function has predefined colors and allows for changing alignment of children.

  Args:
    key (str): The key of the element to set the style for.
    color (Literal["red", "green", "yellow"]): The style to set the element to.
  """

  # Done with string & replace and not f-string, because the style contains curly brackets
  style_string = """
    <style>
      .st-key-XXX {
        background-color: YYY;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        align-items: ZZZ;
  </style>
  """
  
  # Set the given key
  style_string = style_string.replace("XXX", key)

  # Set the alignment
  style_string = style_string.replace("ZZZ", align)
  
  # Set the background color
  if style == "red":
      style_string = style_string.replace("YYY", "rgba(255, 0, 0, 0.7)")
  elif style == "green":
      style_string = style_string.replace("YYY", "rgba(0, 255, 0, 0.7)")
  elif style == "yellow":
      style_string = style_string.replace("YYY", "rgba(255, 255, 0, 0.7)")
  elif style == "grey":
      style_string = style_string.replace("YYY", "rgba(181, 200, 232, 0.7)")
  else:
      raise ValueError(f"Invalid style: {style}")

  st.markdown(style_string, unsafe_allow_html=True)
   


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
      <div id="widget_input_file"></div>
      <div id="image_display"></div>
      <div id="prediction_results"></div>
    </div>
  </div>
</section>
""", unsafe_allow_html=True)



if "image" not in st.session_state:
  st.session_state["image"] = None


def convert_png_to_jpg(image):
  if image.format == "PNG":
    image = image.convert("RGB")
  return image


def process_image(image: Image):
  """
  Process an image using the model.
  If sucessful, calls function to display the results to the user.

  Args:
  image (PIL.Image): The image to process
  """
  try:
    # Make a prediction
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # Get the predicted class
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    # Show the results to the user
    show_results(image, logits, predicted_class)
  
  except Exception as e:
    st.error(f"Error processing the image by the AI model: {e}")



def show_results(image, logits, predicted_class):
  """
  Shows the results of the prediction to the user in Streamlit.

  Args:
  image (PIL.Image): The image that was processed.
  logits (torch.Tensor): The logits from the model.
  predicted_class(str): The predicted class from the logits.
  """
  # Display the image in Streamlit
  st.image(image, caption="Loaded Image", width=400)
  
  # Display the prediction
  st.markdown("<h3 class='text-white text-lg font-medium title-font mb-3'>Prediction:</h3>", unsafe_allow_html=True)
  st.markdown(f"<p>The model predicts this image is: <strong>{predicted_class}</strong></p>", unsafe_allow_html=True)

  # Display confidence scores
  st.markdown("<h3 class='text-white text-lg font-medium title-font mb-3 mt-4'>Confidence Scores:</h3>", unsafe_allow_html=True)
  probs = torch.nn.functional.softmax(logits, dim=1)
  for i, p in enumerate(probs[0]):
      st.markdown(f"<p>{model.config.id2label[i]}: {p.item():.2%}</p>", unsafe_allow_html=True)


def handle_file_upload(file: BytesIO) -> None:
  """
  Handles a file uploaded by the user. Image will be
  loaded as an image and saved into the session state. If the file is a PNG
  image, it will be converted to a JPEG.
  This function should be called only after the user presses the submit button.

  Args:
  file (UploadedFile extending BytesIO): The file uploaded by the user.
  """
  if file is not None:
    try:
      st.session_state["image"] = Image.open(uploaded_file) # Save the loaded image into session state
      st.session_state["image"] = convert_png_to_jpg(st.session_state["image"])
      st.success(f"File '{file.name}' uploaded successfully!")
    except:
      st.error("Error loading the image")

def handle_url_input(url: str) -> None:
  """
  Takes the url of the image as an input. Loads the image from the web
  and stores it as a session state. If the file is png, converts it to jpg.
  This function should be called only after the user presses the submit button.

  Args:
  url (str): The URL of the image to load
  """
  try:
    # Load image from web
    url = url.strip() # Remove whitespace from the url
    response = requests.get(url) # Fetch the image from the URL
    response.raise_for_status() # Raise error if the request fails
    
    # Load the image into memory, converts to jpg
    st.session_state["image"] = Image.open(BytesIO(response.content)) # Saves image into memory using IO and loads it as an image object
    st.session_state["image"] = convert_png_to_jpg(st.session_state["image"])

    # Inform the user that the image was loaded
    st.success(f"Image loaded from URL: {url}")

  except requests.exceptions.RequestException as e:
      st.error(f"Error fetching the image: {e}")
  except Exception as e:
      st.error(f"Error loading the image. Make sure that the URL provided is direct link to the image, not just the website containing the image.")
    



# Instructions for the user
st.text("You can submit your image by uploading a file or by providing a direct URL to it. After inserting the image, press the submit button.")

# Create two tabs (submenu)
with st.container(key="input_container"):
  tab1, tab2 = st.tabs(["Upload File", "Provide Link"])

  # Tab 1: File Upload
  with tab1:
      # Instructions for user
      st.markdown('<div style="color: black;">Please submit your image by selecting the file from your file system. You can also use the drag-and-drop feature.<br />Supported formats: .jpg, .jpeg, .png, .webp</div>', unsafe_allow_html=True)
      
      # User input - upload file widget
      uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], key="widget_input_file", accept_multiple_files=False)
      
      # When sumbit button is pressed
      if st.button("Submit file"):
        st.session_state["image"] = None
        
        # Error handling - no file selected or unsupported file type
        if uploaded_file is None:
          st.error("No file was selected")
        elif uploaded_file.type not in ["image/png", "image/jpg", "image/jpeg", "image/webp"]:
          st.error("Unsupported file type")
        
        # If valid file selected, pass it to convert to image
        else:
          handle_file_upload(uploaded_file)
          
  # Tab 2: Url Input
  with tab2:
      # Instructions for user
      st.markdown('<div style="color: black;">Please submit your image by providing a direct URL to it.<br />Make sure the URL links directly to the image file (e.g., ending in .jpg, .png, or similar).<br />Also make sure that the URL scheme is specified. The URL should start with "http://" or "https://".</div>', unsafe_allow_html=True)
      
      # User input - text field for url
      text_input = st.text_input("", key="widget_input_url", placeholder="https://example.com/image.jpg")
      
      # When sumbit button is pressed, pass it to convert to image
      if st.button("Submit image via link"):
        st.session_state["image"] = None
        handle_url_input(text_input)

# Set CSS style of the input menu (tabs)
set_color("input_container", "grey")


# Runs if an image has been uploaded and loaded
if st.session_state["image"] is not None:
  # Runs image trough AI mode, shows the image preview, shows the AI prediction
  process_image(st.session_state["image"]) 

