import io

import base64

import os

import json

import numpy as np

import cv2

import tensorflow as tf

from flask import Flask, request, jsonify, send_file

from flask_cors import CORS

from tensorflow.keras.preprocessing import image

from PIL import Image



app = Flask(__name__)

CORS(app)  # Enable cross-origin requests for your HTML frontend



# --- 1. DYNAMIC PATH DETECTION ---

# Automatically targets: C:\Users\aravi\Desktop\AgriVision_AI

desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')

BASE_DIR = os.path.join(desktop, 'AgriVision_AI')



MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.keras')

LABELS_PATH = os.path.join(BASE_DIR, 'class_indices.json')

IMG_SIZE = (224, 224)



# --- 2. LOAD ASSETS & LAYER DETECTION ---

last_conv_layer_name = None



if not os.path.exists(MODEL_PATH):

    print(f"❌ FATAL ERROR: Model not found at {MODEL_PATH}")

    model = None

else:

    # Load your trained model

    model = tf.keras.models.load_model(MODEL_PATH)

   

    # Keras 3 Fix: Automatically find the last convolutional layer name

    for layer in reversed(model.layers):

        if "Conv" in layer.__class__.__name__:

            last_conv_layer_name = layer.name

            break

           

    if not last_conv_layer_name:

        last_conv_layer_name = "conv5_block3_out" # Fallback

       

    print(f"✅ Model loaded successfully from Desktop.")

    print(f"✅ Using layer '{last_conv_layer_name}' for Grad-CAM.")



# Load Labels dynamically

if os.path.exists(LABELS_PATH):

    with open(LABELS_PATH, 'r') as f:

        indices = json.load(f)

    # Sort by value (0, 1, 2...) to ensure correct mapping

    class_names = [k for k, v in sorted(indices.items(), key=lambda item: item[1])]

    print(f"✅ Loaded {len(class_names)} plant classes.")

else:

    class_names = ["Unknown"] * 38

    print(f"⚠️ Warning: {LABELS_PATH} not found.")



# --- 3. HELPER FUNCTIONS ---

def preprocess_img(img):

    img = img.resize(IMG_SIZE)

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    return img_array



def make_gradcam_heatmap(img_array, model, layer_name):

    grad_model = tf.keras.models.Model(

        [model.inputs], [model.get_layer(layer_name).output, model.output]

    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]



    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

   

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()



def overlay_gradcam(img, heatmap, alpha=0.4):

    img_np = np.array(img)

    # Resize heatmap to match original image dimensions

    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

   

    # Convert RGB to BGR for OpenCV overlay, then back to RGB

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    overlayed = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)

    overlayed_rgb = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)

    return overlayed_rgb



# --- 4. ROUTES ---

@app.route("/predict", methods=["POST"])

def predict():

    try:

        if "file" not in request.files:

            return jsonify({"error": "No file uploaded"}), 400



        file = request.files["file"]

        img = Image.open(file).convert("RGB")

        img_array = preprocess_img(img)



        # 🚀 1. Prediction

        preds = model.predict(img_array)

        idx = np.argmax(preds[0])

        confidence = float(np.max(preds[0]))

       

        disease_raw = class_names[idx]

        display_name = disease_raw.replace("___", " - ").replace("_", " ")



        # 🚀 2. Grad-CAM Generation

        try:

            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

            overlayed_img = overlay_gradcam(img, heatmap)

           

            # Convert highlighted image to Base64

            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR))

            img_base64 = base64.b64encode(buffer).decode('utf-8')

        except Exception as e:

            print("⚠️ Grad-CAM error:", e)

            img_base64 = None



        # 🚀 3. Response Construction

        severity = "HEALTHY" if "healthy" in display_name.lower() else ("CRITICAL" if confidence > 0.85 else "MODERATE")



        return jsonify({

            "disease": display_name,

            "severity": severity,

            "confidence": f"{confidence*100:.2f}%",

            "pesticide": f"Standard treatment for {display_name}",

            "action": "Ensure proper leaf dry time and isolate infected plants.",

            "gradcam": f"data:image/jpeg;base64,{img_base64}" if img_base64 else None

        })



    except Exception as e:

        return jsonify({"error": str(e)}), 500



# Run app

if __name__ == "__main__":

    app.run(host='0.0.0.0', port=5000, debug=True)