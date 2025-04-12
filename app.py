'''import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model, tokenizer, and processed tracks
model = load_model('sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
tracks = pd.read_csv('processed_tracks.csv')

# Streamlit UI
st.title("Emotion-Based Music Recommendation Chatbot")

# Chat interface
chat_history = st.text_area("Chat with the bot", height=200)
if st.button("Analyze Sentiment"):
    if chat_history:
        chat_seq = tokenizer.texts_to_sequences([re.sub(r'[^\w\s]', '', chat_history.lower())])
        chat_padded = pad_sequences(chat_seq, maxlen=100)
        user_emotion_pred = model.predict(chat_padded)
        user_emotion = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}[np.argmax(user_emotion_pred)]
        st.write(f"Detected emotion: {user_emotion}")

        # Recommend songs
        recommended_songs = tracks[tracks['emotion'] == user_emotion][['artist_name', 'track_name', 'genre']].head(5)
        st.write("Recommended Songs:")
        st.table(recommended_songs)

# Display segregated tracks
st.subheader("Tracks by Emotion")
emotion_filter = st.selectbox("Select Emotion", ['sadness','joy','love','anger','fear','surprise'])
filtered_tracks = tracks[tracks['emotion'] == emotion_filter][['artist_name', 'track_name', 'genre']].head(10)
st.table(filtered_tracks)
'''
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model, tokenizer, and processed tracks
model = load_model('sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
tracks = pd.read_csv('processed_tracks.csv')

# Streamlit UI
st.title("Emotion-Based Music Recommendation Chatbot")

# Initialize session state for conversation
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'responses' not in st.session_state:
    st.session_state.responses = []

# Generic questions
questions = [
    "How has your day been so far?",
    "What’s on your mind today?",
    "Are you feeling any particular way right now?"
]

if st.session_state.step < len(questions):
    st.write(questions[st.session_state.step])
    user_response = st.text_input("Your answer", key=f"q{st.session_state.step}")
    if st.button("Next") and user_response:
        st.session_state.responses.append(user_response)
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == len(questions):
    st.write("Thanks for sharing! Let’s analyze your mood.")
    conversation_text = " ".join(st.session_state.responses)
    
    chat_seq = tokenizer.texts_to_sequences([re.sub(r'[^\w\s]', '', conversation_text.lower())])
    chat_padded = pad_sequences(chat_seq, maxlen=100)
    user_emotion_pred = model.predict(chat_padded)
    user_emotion = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}[np.argmax(user_emotion_pred)]
    st.write(f"Detected emotion: {user_emotion}")

    # Recommend songs
    recommended_songs = tracks[tracks['emotion'] == user_emotion][['artist_name', 'track_name', 'genre']].head(5)
    st.write("Recommended Songs:")
    st.table(recommended_songs)

    st.session_state.step += 1

# Display segregated tracks
if st.session_state.step > len(questions):
    st.subheader("Tracks by Emotion")
    emotion_filter = st.selectbox("Select Emotion", ['sadness','joy','love','anger','fear','surprise'])
    filtered_tracks = tracks[tracks['emotion'] == emotion_filter][['artist_name', 'track_name', 'genre']].head(10)
    st.table(filtered_tracks)