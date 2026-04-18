import numpy as np
import tensorflow as tf
from tensorflow import keras

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([0,0,0,1], dtype=np.float32)  # AND gate

model = keras.Sequential([
    keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
])

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X, y, epochs=5000, batch_size=4, verbose=0)

# Print final loss
print(f"Final loss: {history.history['loss'][-1]:.4f}")

# Predictions
probs = model.predict(X, verbose=0)
print(probs)