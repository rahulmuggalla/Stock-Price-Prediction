from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from pycaret.regression import *

app = Flask(__name__)

def download_stock_data(stock_name):
    # Download stock data from yfinance
    stock_data = yf.download(stock_name)
    return stock_data

def preprocess_stock_data(stock_data):
    # Preprocess stock data
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    return stock_data

def train_best_model(stock_data):
    # Initialize PyCaret regression setup
    reg = setup(data=stock_data, target='Close', train_size=0.8, session_id=123)

    # Compare and select the best ML model
    best_model = compare_models(sort='R2')
    return best_model

def predict_stock_price(stock_data, best_model):
    # Predict stock prices using the trained ML model
    predictions = predict_model(best_model, data=stock_data)
    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_name = request.form['stock_name']

    # Download stock data
    stock_data = download_stock_data(stock_name)

    # Preprocess stock data
    stock_data = preprocess_stock_data(stock_data)

    # Train the best ML model
    best_model = train_best_model(stock_data)

    # Predict stock prices
    predictions = predict_stock_price(stock_data, best_model)

    # Get predicted stock prices for the next 1 week
    predicted_prices = predictions.tail(7)['Close']

    return render_template('result.html', stock_name=stock_name, predicted_prices=predicted_prices)


if __name__ == '__main__':
    app.run(debug=True)