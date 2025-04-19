import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import re
import http.client
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences


SHAZAM_API_HOST = "shazam.p.rapidapi.com"
SHAZAM_API_KEY = "632197b05bmsh9a0974bb39f36edp13dce8jsna88ce1a25d2c"


model = load_model('sentiment_model3.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
tracks = pd.read_csv('processed_tracks.csv')

# Function to get Shazam track link (for fallback or new searches)
def get_shazam_track_link(artist_id):
    try:
        conn = http.client.HTTPSConnection(SHAZAM_API_HOST)
        headers = {
            'x-rapidapi-key': SHAZAM_API_KEY,
            'x-rapidapi-host': SHAZAM_API_HOST
        }
        conn.request("GET", f"/artists/get-latest-release?id={artist_id}&l=en-US", headers=headers)
        res = conn.getresponse()
        data = res.read()
        response = json.loads(data.decode("utf-8"))
        if 'data' in response and response['data']:
            return response['data'][0].get('url', "Link not found")  # Adjust based on API response
        return "Link not found"
    except Exception as e:
        st.write(f"Error fetching link: {e}")
        return "Link not found"

# Streamlit UI
st.title("Music Buddy Chatbot")
st.write("Hey there! I'm your music buddy. Let’s chat a bit, and I’ll find some tunes to match your vibe!")

# Initialize session state for conversation
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'emotion' not in st.session_state:
    st.session_state.emotion = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Dynamic questions based on context
questions = {
    "general": [
        "What’s been on your mind today?",
        "Any cool or tough moments you’d like to share?",
        "How are you feeling about things right now?",
        "Is there something that’s been exciting or bothering you lately?",
        "Anything else you want to tell me about your day?"
    ]
}

# Chat logic
if st.session_state.step == 0 and not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": questions["general"][st.session_state.step]})
    with st.chat_message("assistant"):
        st.write(questions["general"][st.session_state.step])

if prompt := st.chat_input("Your thoughts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.step < len(questions["general"]):
        st.session_state.responses.append(prompt)
        st.session_state.step += 1
        if st.session_state.step < len(questions["general"]):
            with st.chat_message("assistant"):
                st.write(questions["general"][st.session_state.step])
            st.session_state.messages.append({"role": "assistant", "content": questions["general"][st.session_state.step]})
        else:
# Analyze sentiment after all questions
            conversation_text = " ".join(st.session_state.responses)
            chat_seq = tokenizer.texts_to_sequences([re.sub(r'[^\w\s]', '', conversation_text.lower())])
            chat_padded = pad_sequences(chat_seq, maxlen=100)
            user_emotion_pred = model.predict(chat_padded)[0]  # Get prediction array
            emotion_scores = {
                'sadness': user_emotion_pred[0],
                'joy': user_emotion_pred[1],
                'love': user_emotion_pred[2],
                'anger': user_emotion_pred[3],
                'fear': user_emotion_pred[4],
                'surprise': user_emotion_pred[5]
            }
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            st.session_state.emotion = dominant_emotion

            with st.chat_message("assistant"):
                st.write(f"Cool, I think you might be feeling {dominant_emotion} based on our chat. How about some songs to match that?")
                recommended_songs = tracks[tracks['emotion'] == dominant_emotion][['artist_name', 'track_name', 'genre', 'shazam_link']].head(5)
# Re-fetch links if needed
                recommended_songs['shazam_link'] = recommended_songs.apply(lambda row: get_shazam_track_link(str(row['artist_name']).replace(" ", "_")) if row['shazam_link'] == "Link not found" else row['shazam_link'], axis=1)
                st.write("Here are some tunes for you:")
                st.table(recommended_songs)
            st.session_state.messages.append({"role": "assistant", "content": f"Suggested songs for {dominant_emotion}"})
    else:
        with st.chat_message("assistant"):
            st.write("Looks like we’ve covered enough! Let me suggest some songs based on what you shared.")

# Explore more tracks
if st.session_state.emotion:
    st.subheader("Want to Explore More?")
    emotion_filter = st.selectbox("Pick an emotion", ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], index=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'].index(st.session_state.emotion))
    filtered_tracks = tracks[tracks['emotion'] == emotion_filter][['artist_name', 'track_name', 'genre', 'shazam_link']].head(10)
    filtered_tracks['shazam_link'] = filtered_tracks.apply(lambda row: get_shazam_track_link(str(row['artist_name']).replace(" ", "_")) if row['shazam_link'] == "Link not found" else row['shazam_link'], axis=1)
    st.table(filtered_tracks)