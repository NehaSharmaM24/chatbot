import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import pickle

# Load and preprocess emotions dataset
emotions_data = pd.read_csv('emotions.csv').head(70000)  # Load only the first 50,000 rows
emotions_data['text'] = emotions_data['text'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(emotions_data['text'])
sequences = tokenizer.texts_to_sequences(emotions_data['text'])
X = pad_sequences(sequences, maxlen=100)
y = pd.get_dummies(emotions_data['label']).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model with regularization and dropout
model = Sequential()
model.add(Embedding(5000, 64, input_length=100))  # Reduced embedding dimension
model.add(Dropout(0.3))  # Dropout to prevent overfitting
model.add(Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01))))  # Reduced units, added L2
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(16, kernel_regularizer=l2(0.01))))  # Reduced units
model.add(Dropout(0.3))
model.add(Dense(6, activation='softmax', kernel_regularizer=l2(0.01)))  # Added L2 to output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,  # Stop if no improvement for 2 epochs
    restore_best_weights=True
)

# Train the model with early stopping
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,  # Increased max epochs but controlled by early stopping
    batch_size=64,  # Increased batch size to reduce overfitting
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model to check accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Save model and tokenizer
model.save('sentiment_model1.h5')
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)