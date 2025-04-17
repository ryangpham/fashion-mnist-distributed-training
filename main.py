from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
import io

from distributed_training import SimpleCNN

app = FastAPI()

# load model
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# image processing pipeline (need 28x28 images to feed to the model)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = None
    try:
        # read image file from request
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # transform to tensor so model can read it
        img_tensor = transform(image).unsqueeze(0)  # add batch dimension because model expects batches

        # make prediction
        with torch.no_grad():
            output = model(img_tensor)
            predicted = torch.argmax(output, dim=1).item()

        return JSONResponse(content={"predicted_class": predicted})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if image:
            image.close()