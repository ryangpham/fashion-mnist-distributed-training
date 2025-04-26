import json
import torch
import torch.nn as nn
import numpy as np
import io
from PIL import Image
import base64
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def init():
    global model
    
    # Get the path to the model file
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pth')
    
    # Load the model
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
def preprocess_image(image_data):
    """
    Preprocess the image to match Fashion-MNIST format:
    - Convert to grayscale if not already
    - Resize to 28x28
    - Normalize pixel values to [0, 1]
    - Convert to PyTorch tensor with proper dimensions
    """
    # Convert image data to PIL Image
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image).astype('float32') / 255.0
    
    # Reshape to match the input shape expected by the model (B, C, H, W)
    tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    
    return tensor

def run(raw_data):
    try:
        # Parse the input
        input_data = json.loads(raw_data)
        
        # Get the image data
        image_data = None
        
        # Check if input is base64 encoded
        if 'image' in input_data:
            image_data = base64.b64decode(input_data['image'])
        else:
            return json.dumps({"error": "No image data found in request"})
        
        # Preprocess the image
        input_tensor = preprocess_image(image_data)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            
        # Get class probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get the predicted class
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_class_idx]
        
        # Create response with class probabilities
        response = {
            'predicted_class': predicted_class,
            'class_index': predicted_class_idx,
            'probabilities': {class_names[i]: probabilities[i].item() for i in range(len(class_names))}
        }
        
        return json.dumps(response)
    
    except Exception as e:
        return json.dumps({"error": str(e)})