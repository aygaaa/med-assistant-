from sklearn.ensemble import RandomForestClassifier
import joblib

# Assuming the model was trained and exists
diabetes_model = RandomForestClassifier()  # Replace with the actual model
joblib.dump(diabetes_model, "diabetes_model_compatible.sav")
