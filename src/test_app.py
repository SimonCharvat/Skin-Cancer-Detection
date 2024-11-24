import pytest
from app import load_model
from PIL import Image
import torch
import io

@pytest.fixture
def model_and_processor():
    return load_model()

def test_model_loading(model_and_processor):
    processor, model = model_and_processor
    assert processor is not None, "Processor should not be None"
    assert model is not None, "Model should not be None"

def test_image_processing(model_and_processor):
    processor, _ = model_and_processor
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (100, 100))
    
    # Process the image
    inputs = processor(images=dummy_image, return_tensors="pt")
    
    assert 'pixel_values' in inputs, "Processed image should contain 'pixel_values'"
    assert inputs['pixel_values'].shape == (1, 3, 224, 224), "Processed image should have shape (1, 3, 224, 224)"

def test_model_prediction(model_and_processor):
    processor, model = model_and_processor
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (100, 100))
    
    # Process the image and make a prediction
    inputs = processor(images=dummy_image, return_tensors="pt")
    outputs = model(**inputs)
    
    assert outputs.logits.shape == (1, len(model.config.id2label)), "Output should have correct number of classes"
    
    # Test if we can get a predicted class
    predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
    assert predicted_class_idx in model.config.id2label, "Predicted class index should be valid"

def test_image_resizing():
    # Create a dummy image
    original_image = Image.new('RGB', (500, 500))
    
    # Resize the image
    max_size = (300, 300)
    original_image.thumbnail(max_size)
    
    assert original_image.size[0] <= max_size[0] and original_image.size[1] <= max_size[1], "Image should be resized to fit within max_size"

# You might want to add more tests here for other functions in your app

