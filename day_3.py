import tensorflow as tf
import numpy as np
from tensorflow import keras

def salary_model(y_new):
    # Create the training data
    xs = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
    ys = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], dtype=float)
    
    # Define the model
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    
    # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')
    
    # Train the model
    model.fit(xs, ys, epochs=50)
    
    # Ensure input for prediction is a 2D array
    y_new = np.array([[y_new]])  # Convert to a 2D array
    
    # Predict for the reshaped input
    return model.predict(y_new)[0][0]

prediction = salary_model(9.0)
print(prediction)
