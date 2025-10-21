import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Quick stock prediction demo with LSTM
# Install: pip install tensorflow pandas numpy matplotlib scikit-learn

class StockPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
    
    def create_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def prepare_data(self, prices):
        """Prepare data for training"""
        scaled_data = self.scaler.fit_transform(prices.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """Train the model"""
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        self.model = self.create_model((X_train.shape[1], 1))
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    
    def predict(self, X_test):
        """Make predictions"""
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predictions = self.model.predict(X_test, verbose=0)
        return self.scaler.inverse_transform(predictions)

def demo():
    print('Stock Prediction Demo - Ready in under 2 mins!')
    
    # Generate sample stock data
    np.random.seed(42)
    days = 300
    prices = 100 + np.cumsum(np.random.randn(days) * 2)
    
    predictor = StockPredictor(sequence_length=60)
    X, y = predictor.prepare_data(prices)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train
    print('Training LSTM model...')
    predictor.train(X_train, y_train, epochs=5, batch_size=16)
    
    # Predict
    predictions = predictor.predict(X_test)
    
    print(f'Predictions shape: {predictions.shape}')
    print('Sample predictions:', predictions[:5].flatten())
    print('\nModel trained successfully!')
    print('Replace with actual stock data for real predictions.')

if __name__ == '__main__':
    demo()
