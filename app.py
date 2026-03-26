from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    # Decode the base64 image
    header, encoded = data.split(",", 1)
    data = base64.b64decode(encoded)
    
    # Convert to OpenCV format
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.imread_color)

    # --- YOUR MODEL WORK HERE ---
    # 1. Preprocess frame
    # 2. Pass to CNN for feature extraction
    # 3. Pass sequence to RNN/LSTM
    # prediction = predict_sign(frame)
    
    prediction = "HELLO" # Placeholder result

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)