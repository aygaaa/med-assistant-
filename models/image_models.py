from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

app = Flask(__name__)
CORS(app)

# Pneumonia Model Setup
pneumonia_class_names = ["COVID-19", "Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"]
pneumonia_model = models.resnet18()
pneumonia_model.fc = torch.nn.Linear(pneumonia_model.fc.in_features, len(pneumonia_class_names))
pneumonia_model.load_state_dict(torch.load("resnet18_xray_model.pth", map_location=torch.device("cpu")))
pneumonia_model.eval()

# Skin Disease Model Setup
skin_class_names = [
    " Warts Molluscum and other Viral Infections ",
    " Melanoma ",
    " Basal Cell Carcinoma ",
    " Benign Keratosis-like Lesions ",
    " Psoriasis pictures Lichen Planus and related diseases ",
]
skin_model = models.resnet18()
skin_model.fc = torch.nn.Linear(512, len(skin_class_names))
skin_model.load_state_dict(torch.load("resnet18_skin_disease_model.pth", map_location=torch.device("cpu")))
skin_model.eval()

# Shared Image Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Prediction Endpoint
@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Load and preprocess the image
    image_file = request.files["image"]
    image = Image.open(image_file.stream).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Select the model
    if model_name == "pneumonia":
        model = pneumonia_model
        class_names = pneumonia_class_names
    elif model_name == "skin":
        model = skin_model
        class_names = skin_class_names
    else:
        return jsonify({"error": "Invalid model name"}), 400

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return jsonify({"prediction": predicted_class})


if __name__ == "__main__":
    app.run(debug=True)
    
