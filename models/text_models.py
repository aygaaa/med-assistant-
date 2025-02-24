from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the model
try:
    loaded_model = pickle.load(open("diabetes_model.sav", "rb"))
except FileNotFoundError:
    raise FileNotFoundError("Error: The model file 'diabetes_model.sav' was not found.")
except Exception as e:
    raise Exception(f"An error occurred while loading the model: {e}")

# Define the mapping rules
mappings = {
    "gender": {"Male": 0, "Female": 1},
    "hypertension": {"yes": 1, "no": 0, "y": 1, "n": 0},
    "heart_disease": {"yes": 1, "no": 0, "y": 1, "n": 0},
    "smoking_history": {"yes": 1, "no": 0, "y": 1, "n": 0},
}

# Function to map input data
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

# Define the prediction route
@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    try:
        # Get JSON data from the request
        data = request.json

        # Validate required fields
        required_fields = [
            "gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "hba1c_level", "blood_glucose_level"
        ]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing one or more required fields"}), 400

        # Map the input data
        mapped_data = map_input_data(data)

        # Convert to NumPy array and reshape for prediction
        input_data_as_numpy_array = np.asarray(mapped_data).reshape(1, -1)

        # Make the prediction
        prediction = loaded_model.predict(input_data_as_numpy_array)
        result = "diabetic" if prediction[0] else "not diabetic"

        # Return the result as JSON
        return jsonify({"prediction": result})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(port=5002, debug=True)