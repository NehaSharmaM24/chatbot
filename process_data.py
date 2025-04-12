import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model and tokenizer
model = load_model('sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load and process tracks.csv
tracks = pd.read_csv('tracks.csv')
tracks['lyrics'] = tracks['lyrics'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
tracks_sequences = tokenizer.texts_to_sequences(tracks['lyrics'])
tracks_X = pad_sequences(tracks_sequences, maxlen=100)
tracks_predictions = model.predict(tracks_X)
emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
tracks['emotion'] = np.argmax(tracks_predictions, axis=1)
tracks['emotion'] = tracks['emotion'].map(emotion_map)

# Save processed tracks
tracks.to_csv('processed_tracks.csv', index=False)