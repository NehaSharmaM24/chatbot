import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
from tqdm import tqdm
import time

# Shazam API
RAPIDAPI_KEY = "632197b05bmsh9a0974bb39f36edp13dce8jsna88ce1a25d2c"

# Load model and tokenizer
model = load_model('sentiment_model3.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to get Shazam track link using search endpoint
def get_song_link(artist: str, track: str):
    url = "https://shazam.p.rapidapi.com/search"
    query = f"{artist} {track}"
    
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "shazam.p.rapidapi.com"
    }

    params = {
        "term": query,
        "locale": "en-US",
        "offset": "0",
        "limit": "1"
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return "Link not found"

    data = response.json()
    try:
        song = data["tracks"]["hits"][0]["track"]
        title = song["title"]
        subtitle = song["subtitle"]
        song_url = song["url"]
        print(f"Found: {title} by {subtitle}")
        print(f"Song URL: {song_url}")
        return song_url
    except (KeyError, IndexError):
        print("Song not found.")
        return "Link not found"

# Load and process a subset of tracks.csv - 500 due to reuqest limits
tracks = pd.read_csv('tracks.csv').head(500)
tracks['lyrics'] = tracks['lyrics'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
tracks_sequences = tokenizer.texts_to_sequences(tracks['lyrics'])
tracks_X = pad_sequences(tracks_sequences, maxlen=100)
tracks_predictions = model.predict(tracks_X)
emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
tracks['emotion'] = np.argmax(tracks_predictions, axis=1)
tracks['emotion'] = tracks['emotion'].map(emotion_map)

# Add Shazam links with progress bar
tracks['shazam_link'] = [get_song_link(row.artist_name, row.track_name) for row in tqdm(tracks.itertuples(), total=len(tracks))]
time.sleep(1)

#Saving the final csv file
tracks.to_csv('processed_tracks.csv', index=False)
print("Processed tracks with Shazam links (subset of 30) saved to processed_tracks.csv")