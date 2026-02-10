import tensorflow as tf
import numpy as np
import time

print("TF Version:", tf.__version__)

# Create dummy data
X = np.random.random((100, 60, 20))
y = np.random.randint(0, 2, 100)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(60, 20)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

print("Starting training...")
start = time.time()
model.fit(X, y, epochs=5, verbose=1)
print(f"Training took {time.time() - start:.2f}s")
