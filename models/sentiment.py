from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = Flask(__name__)
CORS(app)

# Load the model and tokenizer
model_path = "roberta_emotion_model.pth"
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=13)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define emotion classes
classes = [
    'empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise',
    'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger'
]

@app.route("/predict/sentiment", methods=["POST"])
def predict_sentiment():
    try:
        data = request.json
        sentence = data.get("sentence", "")
        
        if not sentence:
            return jsonify({"error": "Missing or empty 'sentence' field"}), 400
        
        # Tokenize input and make predictions
        encodings = tokenizer([sentence], truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encodings)
        prediction = outputs.logits.argmax(dim=-1).item()
        predicted_emotion = classes[prediction]
        
        return jsonify({"emotion": predicted_emotion})  # Updated key to 'emotion'
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)
