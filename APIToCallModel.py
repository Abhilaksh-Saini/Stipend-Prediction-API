from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

# Configure CORS properly
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model and encoders
model = joblib.load("stipend_prediction_model.joblib")
encoder_city = joblib.load("encoder_city.joblib")
encoder_sector = joblib.load("encoder_sector.joblib")

@app.route('/')
def home():
    return "Stipend Prediction API is running!"

def predict_stipend(city, sector, duration, remote):
    """Helper function to process input and predict stipend."""
    city_encoded = encoder_city.transform([city])[0]
    sector_encoded = encoder_sector.transform([sector])[0]
    remote_encoded = 1 if remote.lower() == "yes" else 0
    
    input_data = pd.DataFrame([[city_encoded, sector_encoded, duration, remote_encoded]],
                              columns=["City", "Sector", "Duration_Months", "Remote"])
    
    predicted_stipend = model.predict(input_data)[0]
    return round(predicted_stipend, 2)

@app.route('/predict', methods=['POST', 'OPTIONS', 'GET'])
def predict():
    try:
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = jsonify({"message": "CORS preflight request successful"})
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
            return response, 200

        # Process the actual request
        if request.method == 'POST':
            data = request.get_json()
        elif request.method == 'GET':
            data = request.args
        
        city = data["City"]
        sector = data["Sector"]
        duration = int(data["Duration_Months"])
        remote = data["Remote"]

        # Get prediction
        stipend = predict_stipend(city, sector, duration, remote)

        response = jsonify({"Predicted Stipend": stipend})
        response.headers.add("Access-Control-Allow-Origin", "*")  # Add CORS header to response
        return response
    
    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "*")  # Add CORS header to error response
        return response, 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
