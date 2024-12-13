from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the model and mapping file
def load_mapping_file(mapping_file):
    with open(mapping_file, 'r') as f:
        return {int(line.split()[0]): chr(int(line.split()[1])) for line in f}

def load_resources():
    letter_model = load_model('models/emnist_letter_recognition_model.h5')
    digit_model = load_model('models/digit_recognition_model.h5')

    letter_mapping = load_mapping_file('models/letter_mapping.txt')
    digit_mapping = load_mapping_file('models/digit_mapping.txt')

    app.logger.info("Models and mappings loaded successfully!")
    return letter_model, digit_model, letter_mapping, digit_mapping

# Preprocess image function
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Invalid image file or unsupported format.")
        if np.mean(image) > 128:
            image = 255 - image
        image_resized = cv2.resize(image, (28, 28))
        image_normalized = image_resized.astype("float32") / 255.0
        return np.expand_dims(image_normalized, axis=(0, -1))
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {str(e)}")

# Predict function
def predict_without_true_label(image_path, is_letter):
    try:
        image_input = preprocess_image(image_path)
        model = letter_model if is_letter else digit_model
        mapping_dict = letter_mapping if is_letter else digit_mapping
        
        app.logger.info(f"Using {'letter' if is_letter else 'digit'} model for prediction.")
        prediction = model.predict(image_input)
        predicted_label = np.argmax(prediction)
        predicted_char = mapping_dict[predicted_label]
        return predicted_char, prediction[0]
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

# API endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files["image"]
        image_path = os.path.join("uploads", image.filename)
        image.save(image_path)

        is_letter_param = request.form.get("is_letter")
        if is_letter_param is None:
            return jsonify({"error": "`is_letter` parameter is required (true/false)."}), 400

        is_letter = is_letter_param.lower() == "true"

        app.logger.info(f"Received is_letter={is_letter}. Using {'letter' if is_letter else 'digit'} model.")

        predicted_char, probabilities = predict_without_true_label(image_path, is_letter)
        response = {
            "predicted_char": predicted_char,
            "probabilities": probabilities.tolist(),
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load models and mappings globally
letter_model, digit_model, letter_mapping, digit_mapping = load_resources()

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(host="0.0.0.0", port=5000)
