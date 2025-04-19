import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import pickle

# Load and preprocess emotions dataset
emotions_data = pd.read_csv('emotions.csv')
emotions_data = emotions_data.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 8333), random_state=42))  # ~50,000 total
emotions_data['text'] = emotions_data['text'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
tokenizer = Tokenizer(num_words=10000)  # Increased vocabulary size
tokenizer.fit_on_texts(emotions_data['text'])
sequences = tokenizer.texts_to_sequences(emotions_data['text'])
X = pad_sequences(sequences, maxlen=150)  # Increased maxlen for more context
y = pd.get_dummies(emotions_data['label']).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model architecture
model = Sequential()
model.add(Embedding(10000, 128, input_length=150))  
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.005)))) 
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32, kernel_regularizer=l2(0.005))))  
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax', kernel_regularizer=l2(0.005)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3, 
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# Model Training
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,  
    batch_size=512,  
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluating the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Saving model and tokenizer
model.save('sentiment_model3.h5')
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)