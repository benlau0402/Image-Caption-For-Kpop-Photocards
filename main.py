import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# Define paths
WORKING_DIR = '/output'
MODEL_PATH = os.path.join(WORKING_DIR, 'best_model.keras')
TOKENIZER_PATH = os.path.join(WORKING_DIR, 'tokenizer.pkl')

# Load the trained model
print("Loading model...")
model = load_model(MODEL_PATH)

# Load the tokenizer
print("Loading tokenizer...")
with open(TOKENIZER_PATH, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Parameters
# Update the max caption length to match your model's expected input shape (expected: 20)
MAX_CAPTION_LENGTH = 20  
VOCAB_SIZE = len(tokenizer.word_index) + 1  # Ensure this matches the training setup

# Function to extract features from an image using the VGG16 model
def extract_features(image_path):
    print(f"Processing image: {image_path}")

    # Load the original VGG16 model (with the top layers)
    vgg_model = VGG16(weights='imagenet')
    # Create a new model that outputs the feature vector from the 'fc2' layer
    new_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.get_layer('fc2').output)

    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)

    # Extract features
    features = new_model.predict(image, verbose=0)
    return features  # Shape should be (1, 4096)

# Function to generate captions
def generate_caption(model, tokenizer, image_features, max_length):
    # Start the caption generation process
    input_text = 'startseq'  # Assumes 'startseq' is the start token
    for i in range(max_length):
        # Tokenize the current sequence
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        # Pad the sequence to the maximum caption length (now 20)
        sequence = np.pad(sequence, (0, max_length - len(sequence)), mode='constant')
        # Predict the next word
        prediction = model.predict([image_features, np.array([sequence])], verbose=0)
        next_word_id = np.argmax(prediction)
        next_word = tokenizer.index_word.get(next_word_id, None)
        if not next_word or next_word == 'endseq':  # End token or no word found
            break
        input_text += f' {next_word}'
    # Remove 'startseq' from the generated caption
    final_caption = input_text.replace('startseq', '').strip()
    return final_caption

# Main script
if __name__ == "__main__":
    # Hardcode the image path
    image_path = '/Users/benlau/Downloads/IMG_4548.JPG'

    # Extract image features
    image_features = extract_features(image_path)

    # Generate caption
    caption = generate_caption(model, tokenizer, image_features, MAX_CAPTION_LENGTH)
    print(f"Generated Caption: {caption}")