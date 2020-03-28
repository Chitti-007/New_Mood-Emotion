import os

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence

from preprocess import load_emotion_data
from model import get_model_emotions
from config import sequence_length, embedding_size, batch_size, epochs
from keras.utils import to_categorical
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test, vocab = load_emotion_data()

vocab_size = len(vocab)
vocab1=np.array(vocab)

print("Vocab size:", vocab_size)

X_train = sequence.pad_sequences(X_train, maxlen=sequence_length)
X_test = sequence.pad_sequences(X_test, maxlen=sequence_length)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)

# np.savetxt("Mood:Emotion Code/data/X_test.csv",X_test,fmt='%1.4d')
# print(len(vocab))


print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

model = get_model_emotions(vocab_size, sequence_length=sequence_length, embedding_size=embedding_size)
model.load_weights("Mood:Emotion Code/results/model_v1_0.68_0.73.h5")
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

if not os.path.isdir("results"):
    os.mkdir("results")

checkpointer = ModelCheckpoint("results/model_v1_{val_loss:.2f}_{val_accuracy:.2f}.h5",save_best_only=True, verbose=1)

history=model.fit(X_train, y_train, epochs=epochs,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            callbacks=[checkpointer])

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()