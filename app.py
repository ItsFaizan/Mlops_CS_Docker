from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('insurance_model.pkl')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        print("Received JSON data:", data)
        
        # Convert data to a pandas DataFrame
        df = pd.DataFrame(data)
        
        # Make prediction
        predictions = model.predict(df)
        print("Predictions:", predictions)
        
        # Return response
        return jsonify({"predicted_charges": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
