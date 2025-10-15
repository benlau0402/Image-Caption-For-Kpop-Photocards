import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import io

# Paths
WORKING_DIR = '/output'
MODEL_PATH = os.path.join(WORKING_DIR, 'best_model.keras')
TOKENIZER_PATH = os.path.join(WORKING_DIR, 'tokenizer.pkl')

print("Loading caption model...")
model = load_model(MODEL_PATH)

print("Loading tokenizer...")
with open(TOKENIZER_PATH, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Constants (should match training)
MAX_CAPTION_LENGTH = 21
VOCAB_SIZE = len(tokenizer.word_index) + 1  # might be capped by num_words during training

# ---- Load VGG16 once (performance) ----
print("Initializing VGG16 feature extractor...")
_base_vgg = VGG16(weights='imagenet')
_vgg_fc2 = Model(inputs=_base_vgg.inputs, outputs=_base_vgg.get_layer('fc2').output)

def extract_features(image_bytes: bytes) -> np.ndarray:
    """Return (1, 4096) float32 feature vector from fc2 layer."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feats = _vgg_fc2.predict(arr, verbose=0)
    return feats.astype('float32', copy=False)

def generate_caption(model, tokenizer, image_features, max_length: int) -> str:
    """Greedy decoding with basic guards."""
    input_text = 'startseq'
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([input_text])[0]
        # pad to the right with zeros; ensure int32 for embedding input
        if len(seq) < max_length:
            seq = np.pad(seq, (0, max_length - len(seq)), mode='constant')
        else:
            seq = np.array(seq[:max_length])  # truncate if somehow longer

        seq = np.asarray(seq, dtype='int32')
        preds = model.predict([image_features, np.array([seq], dtype='int32')], verbose=0)
        next_id = int(np.argmax(preds, axis=-1)[0])

        # Avoid pad (0) or invalid id; also stop on end token
        if next_id == 0:
            break
        next_word = tokenizer.index_word.get(next_id)
        if not next_word or next_word == 'endseq':
            break

        input_text += f' {next_word}'

    # Clean final caption
    final_caption = input_text.replace('startseq', '').strip()
    return final_caption or "(no caption)"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB limit

@app.route('/caption', methods=['POST'])
def caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if not file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
        return jsonify({'error': 'Unsupported file type'}), 400

    try:
        img_bytes = file.read()
        image_features = extract_features(img_bytes) 
        caption = generate_caption(model, tokenizer, image_features, MAX_CAPTION_LENGTH)
        return jsonify({'caption': caption, 'success': True}), 200
    except Exception as e:
        return jsonify({'error': f'Caption generation failed: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)