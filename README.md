# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the historical stock price data (replace 'your_data.csv' with your dataset)
data = pd.read_csv('your_data.csv')

# Take the 'Close' prices as the target variable
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data (scaling to [0, 1])
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# Split the data into training and testing sets
train_size = int(len(prices_scaled) * 0.7)
train_data, test_data = prices_scaled[0:train_size], prices_scaled[train_size:]

# Create sequences of data for training and testing
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # You can adjust this sequence length
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the scaled predictions to get actual stock prices
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Calculate and print the Root Mean Squared Error (RMSE)
train_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_train), train_predictions))
test_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), test_predictions))
print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(train_predictions):], scaler.inverse_transform(y_train), label='Actual Train Prices', color='blue')
plt.plot(data.index[-len(test_predictions):], scaler.inverse_transform(y_test), label='Actual Test Prices', color='green')
plt.plot(data.index[-len(test_predictions):], test_predictions, label='Predicted Test Prices', color='red')
plt.legend()
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
