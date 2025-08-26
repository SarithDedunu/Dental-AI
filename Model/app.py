from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

# It's important to set Matplotlib's backend to Agg to prevent
# display errors when running in a non-GUI environment like Flask.
plt.switch_backend('Agg')

app = Flask(__name__)

# Load the model once when the application starts
try:
    model = load_model('new_efficientnet_dentalAI_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the class names based on your model's output order
class_names = ['Healthy Teeth', 'Unhealthy Teeth']

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, predictions = grad_model(img_array)
        if pred_index is None:
            # predictions is a list of tensors, so we get the first item
            pred_index = tf.argmax(predictions[0][0])
        
        # Access the first item of the predictions list before slicing
        class_channel = predictions[0][:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def generate_heatmap_image(image_bytes, heatmap):
    img = load_img(io.BytesIO(image_bytes))
    img = img_to_array(img)

    # Resize the heatmap to match the original image size
    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps['jet']
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap).resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # Save the image to a byte buffer
    buf = io.BytesIO()
    superimposed_img.save(buf, format='JPEG')
    buf.seek(0)
    return buf

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    image_bytes = file.read()

    # Preprocess image for prediction
    try:
        img = load_img(io.BytesIO(image_bytes), target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
    except Exception as e:
        return jsonify({'error': f"Error processing image: {str(e)}"}), 400

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(predictions[0][class_index])
    predicted_class = class_names[class_index]

    # Generate Grad-CAM heatmap
    last_conv_layer_name = 'conv5_block3_3_conv'
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Generate the heatmap image by passing the original image bytes
    heatmap_image_bytes = generate_heatmap_image(image_bytes, heatmap)

    # Encode the image to base64
    heatmap_base64 = base64.b64encode(heatmap_image_bytes.getvalue()).decode('utf-8')

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': predictions[0].tolist(),
        'heatmap_image': heatmap_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)