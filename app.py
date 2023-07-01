from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('save_model.h5')


@app.route('/')
def appp():
	return "5"

# Route for object detection
@app.route('/detect-object', methods=['POST'])
def detect_pothole():
    # Get the image file from the request
    image_file = request.files['image']

    # Load and preprocess the image
    image = Image.open(image_file)
    image = image.resize((64, 64))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Debug statements
    print('Image shape:', image.shape)
    print('Image data:', image)

    # Make predictions
    result = model.predict(image)

    # Convert the prediction to a label
    if result[0][0] == 1:
        prediction = 'pothole'
    else:
        prediction = 'Normal'

    # Return the prediction as a JSON response
    response = {'prediction': prediction}
    return jsonify(response)

# Run the Flask application
if __name__ == '__main__':
    app.run()
