import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
from tensorflow import keras

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = {
    'AND': np.array([[0],[0],[0],[1]], dtype=np.float32),
    'OR': np.array([[0],[1],[1],[1]], dtype=np.float32),
    'XOR': np.array([[0],[1],[1],[0]], dtype=np.float32),   
}

##### Perceptron #####
modelPerceptron = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model = modelPerceptron
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

for key, labels in y.items():
    print(f"\n-----Training for {key} gate with single perceptron-----")
    history = model.fit(X, labels, epochs=500, batch_size=4, verbose=0)

    # Print final loss
    print(f"Final loss: {history.history['loss'][-1]:.4f}")

    # Predictions
    probs = model.predict(X, verbose=0)
    print(probs)

##### Two Layer NN #####
modelTwoLayerNN = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])

model = modelTwoLayerNN
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

for key, labels in y.items():
    print(f"\n-----Training for {key} gate with two layer NN-----")
    history = model.fit(X, labels, epochs=500, batch_size=4, verbose=0)

    # Print final loss
    print(f"Final loss: {history.history['loss'][-1]:.4f}")

    # Predictions
    probs = model.predict(X, verbose=0)
    print(probs)
