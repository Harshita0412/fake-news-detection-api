from flask import Flask, request, jsonify
import os
import pickle

# Disable GPU/CUDA attempts BEFORE importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# --------------------------------------------------
# TensorFlow CPU configuration
# --------------------------------------------------
tf.config.set_visible_devices([], "GPU")

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MAX_LEN = 300

# --------------------------------------------------
# Load model from Hugging Face
# --------------------------------------------------
print("Downloading/loading model...")

MODEL_PATH = hf_hub_download(
    repo_id="harshitamishra04/fake-news-detection-lstm",
    filename="fake_news_lstm.keras"
)

print("Model downloaded.")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")

# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------
print("Loading tokenizer...")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("Tokenizer loaded successfully.")


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_news(article):

    sequence = tokenizer.texts_to_sequences([article])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probability = float(
        model.predict(
            padded,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:
        prediction = "FAKE"
        confidence = probability
    else:
        prediction = "REAL"
        confidence = 1 - probability

    return prediction, confidence


# --------------------------------------------------
# Home route
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "online",
        "message": "Fake News Detection API is running"
    })


# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


# --------------------------------------------------
# Prediction API
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "Request body must contain JSON."
            }), 400

        if "article" not in data:
            return jsonify({
                "error": "Please provide article text."
            }), 400

        article = data["article"]

        if not isinstance(article, str):
            return jsonify({
                "error": "Article must be a string."
            }), 400

        if not article.strip():
            return jsonify({
                "error": "Article text cannot be empty."
            }), 400

        prediction, confidence = predict_news(article)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Start server
# --------------------------------------------------
if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
