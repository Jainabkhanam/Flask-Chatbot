from flask import Flask, request, jsonify, render_template
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model, tokenizer, and label encoder
model = load_model('chat_model')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Load intents data
with open('intents.json') as file:
    data = json.load(file)

max_len = 20

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json.get('message')
    if not input_data:
        return jsonify({'error': 'Invalid input data'}), 400
    
    # Preprocess input and make prediction
    sequences = tokenizer.texts_to_sequences([input_data])
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
    result = model.predict(padded_sequences)
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    
    for intent in data['intents']:
        if intent['tag'] == tag:
            response = np.random.choice(intent['responses'])
            return jsonify({'response': response})
    
    return jsonify({'response': "I'm not sure how to respond to that."})

if __name__ == '__main__':
    app.run(debug=True)
