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
    header, encoded = data.split(",", 1)
    data = base64.b64decode(encoded)
    
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.imread_color)

    #  MODEL WORK HERE ---
    
    
    prediction = "HELLO" 

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
