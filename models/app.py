from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# -----------------------------
# Sentiment Analysis Setup
# -----------------------------
# Load the model and tokenizer
sentiment_model_path = "roberta_emotion_model.pth"
sentiment_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
sentiment_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=13)
sentiment_model.load_state_dict(torch.load(sentiment_model_path, map_location=torch.device('cpu')))
sentiment_model.eval()

# Define emotion classes
sentiment_classes = [
    'empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise',
    'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger'
]

# -----------------------------
# Pneumonia Model Setup
# -----------------------------
pneumonia_class_names = ["COVID-19", "Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"]
pneumonia_model = models.resnet18()
pneumonia_model.fc = torch.nn.Linear(pneumonia_model.fc.in_features, len(pneumonia_class_names))
pneumonia_model.load_state_dict(torch.load("resnet18_xray_model.pth", map_location=torch.device("cpu")))
pneumonia_model.eval()

# -----------------------------
# Skin Disease Model Setup
# -----------------------------
skin_class_names = [
    "10. Warts Molluscum and other Viral Infections - 2103",
    "2. Melanoma 15.75k",
    "4. Basal Cell Carcinoma (BCC) 3323",
    "6. Benign Keratosis-like Lesions (BKL) 2624",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k",
]
skin_model = models.resnet18()
skin_model.fc = torch.nn.Linear(512, len(skin_class_names))
skin_model.load_state_dict(torch.load("resnet18_skin_disease_model.pth", map_location=torch.device("cpu")))
skin_model.eval()

# Shared Image Transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# Diabetes Model Setup
# -----------------------------
try:
    diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))
except FileNotFoundError:
    raise FileNotFoundError("Error: The model file 'diabetes_model.sav' was not found.")
except Exception as e:
    raise Exception(f"An error occurred while loading the model: {e}")

# Mapping rules for diabetes input
mappings = {
    "gender": {"Male": 0, "Female": 1},
    "hypertension": {"yes": 1, "no": 0, "y": 1, "n": 0},
    "heart_disease": {"yes": 1, "no": 0, "y": 1, "n": 0},
    "smoking_history": {"yes": 1, "no": 0, "y": 1, "n": 0},
}

def map_input_data(data):
    try:
        processed_data = [
            mappings["gender"][data["gender"]],
            float(data["age"]),
            mappings["hypertension"][data["hypertension"].lower()],
            mappings["heart_disease"][data["heart_disease"].lower()],
            mappings["smoking_history"][data["smoking_history"].lower()],
            float(data["bmi"]),
            float(data["hba1c_level"]),
            int(data["blood_glucose_level"]),
        ]
        return processed_data
    except KeyError as e:
        raise ValueError(f"Invalid value provided for: {e}")
    except Exception as e:
        raise ValueError(f"An error occurred during data mapping: {e}")

# -----------------------------
# Routes
# -----------------------------
@app.route("/predict/sentiment", methods=["POST"])
def predict_sentiment():
    try:
        data = request.json
        sentence = data.get("sentence", "")

        if not sentence:
            return jsonify({"error": "Missing or empty 'sentence' field"}), 400

        encodings = sentiment_tokenizer([sentence], truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = sentiment_model(**encodings)
        prediction = outputs.logits.argmax(dim=-1).item()
        predicted_emotion = sentiment_classes[prediction]

        return jsonify({"emotion": predicted_emotion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/<model_name>", methods=["POST"])
def predict_image(model_name):
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file.stream).convert("RGB")
    input_tensor = image_transform(image).unsqueeze(0)

    if model_name == "pneumonia":
        model = pneumonia_model
        class_names = pneumonia_class_names
    elif model_name == "skin":
        model = skin_model
        class_names = skin_class_names
    else:
        return jsonify({"error": "Invalid model name"}), 400

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return jsonify({"prediction": predicted_class})

@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.json

        required_fields = [
            "gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "hba1c_level", "blood_glucose_level"
        ]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing one or more required fields"}), 400

        mapped_data = map_input_data(data)
        input_data_as_numpy_array = np.asarray(mapped_data).reshape(1, -1)

        prediction = diabetes_model.predict(input_data_as_numpy_array)
        result = "diabetic" if prediction[0] else "not diabetic"

        return jsonify({"prediction": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)