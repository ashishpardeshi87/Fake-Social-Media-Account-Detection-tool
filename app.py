from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load pre-trained models
models = {
    'Instagram': joblib.load('Instagram_model.pkl'),
    'Facebook': joblib.load('Facebook_model.pkl'),
    'X': joblib.load('X_model.pkl')
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_model_result', methods=['POST'])
def get_model_result():
    platform = request.json.get('platform')
    
    if platform not in models:
        return jsonify({'result': 'Platform not supported'}), 400
    
    model = models[platform]
    
    # Get user input from the form (simulated here, use actual form data)
    input_data = {
        'Instagram': [1, 1, 100, 5000, 300, 7],  # Example data for Instagram
        'Facebook': [1, 1, 10, 500, 300, 8],  # Example data for Facebook
        'X': [5000, 300, 10000, 8]  # Example data for X
    }

    # Fetch the data based on the platform
    platform_data = input_data.get(platform)
    
    if not platform_data:
        return jsonify({'result': 'No input data found for the selected platform'}), 400
    
    # Prediction based on selected platform and input data
    prediction_probabilities = model.predict_proba([platform_data])  # Predict probabilities
    
    # Get probability for positive class (index 1)
    positive_class_probability = prediction_probabilities[0][1]
    
    # Convert to percentage
    percentage_result = positive_class_probability * 100
    
    return jsonify({'result': f"{percentage_result:.2f}%"})  # Return percentage result

if __name__ == '__main__':
    app.run(debug=True)
