# RNN_Rollrate
# Importing necessary libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load the dataset
df = pd.read_csv("roll_rates.csv", header=None, names=["roll_rates"])

# Define the input and output sequences
input_seq = df.values
output_seq = np.concatenate((input_seq[12:], np.zeros((12,1))), axis=0)

# Define the function to reshape the data into 3D format
def reshape_data(data, n_steps):
    X, y = [], []
        for i in range(len(data)-n_steps):
                X.append(data[i:i+n_steps, :])
                        y.append(data[i+n_steps, :])
                            X = np.array(X)
                                y = np.array(y)
                                    return X, y

                                    # Reshape the input and output sequences into 3D format
                                    n_steps = 12
                                    X, y = reshape_data(input_seq, n_steps)
                                    y = y.reshape(-1, 1)
                                    X = X.reshape(X.shape[0], X.shape[1], 1)

                                    # Define the RNN model
                                    model = Sequential()
                                    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
                                    model.add(Dense(1))
                                    model.compile(optimizer='adam', loss='mse')

                                    # Train the RNN model on the entire dataset
                                    model.fit(X, y, epochs=100, batch_size=16, verbose=0)

                                    # Generate the predictions for the next 12 or 18 months
                                    n_preds = 12 # Change this to 18 if you want to predict for 18 months
                                    preds = input_seq[-n_steps:].reshape(1, n_steps, 1)
                                    for i in range(n_preds):
                                        next_pred = model.predict(preds)[0][0]
                                            preds = np.append(preds[:,1:,:], [[next_pred]], axis=1)

                                            # Print the predicted roll rates
                                            print("Predicted roll rates for the next {} months: {}".format(n_preds, preds.reshape(n_preds)))
                                            