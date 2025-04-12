import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from sklearn.model_selection import train_test_split

# Load and preprocess emotions dataset
emotions_data = pd.read_csv('emotions.csv')
emotions_data['text'] = emotions_data['text'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(emotions_data['text'])
sequences = tokenizer.texts_to_sequences(emotions_data['text'])
X = pad_sequences(sequences, maxlen=100)
y = pd.get_dummies(emotions_data['label']).values

# Train Bi-LSTM model
model = Sequential()
model.add(Embedding(5000, 128, input_length=100))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)

# Save model and tokenizer
model.save('sentiment_model.h5')
import pickle
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)