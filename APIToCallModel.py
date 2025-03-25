from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load the trained model and encoders
model = joblib.load("stipend_prediction_model.joblib")
encoder_city = joblib.load("encoder_city.joblib")
encoder_sector = joblib.load("encoder_sector.joblib")

@app.route('/')
def home():
    return "Stipend Prediction API is running!"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract and preprocess input features
        city = data["City"]
        sector = data["Sector"]
        duration = data["Duration_Months"]
        remote = data["Remote"]

        # Encode categorical variables
        city_encoded = encoder_city.transform([city])[0]
        sector_encoded = encoder_sector.transform([sector])[0]
        remote_encoded = 1 if remote.lower() == "yes" else 0

        # Prepare input data for prediction
        input_data = pd.DataFrame([[city_encoded, sector_encoded, duration, remote_encoded]],
                                  columns=["City", "Sector", "Duration_Months", "Remote"])

        # Make prediction
        predicted_stipend = model.predict(input_data)[0]

        # Return prediction as JSON
        return jsonify({"Predicted Stipend": round(predicted_stipend, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
