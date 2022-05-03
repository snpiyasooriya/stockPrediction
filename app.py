from turtle import shape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, scale


# defining data set date range
start = '2010-01-01'
end = '2022-01-01'

st.title('Stock Market Value Prediction')

# get stock ticket fom user inputs
stock_ticker = st.text_input('Enter Stock Ticker', 'TSLA')

# get data from yahoo using panda_datreader

df = data.DataReader(stock_ticker, "yahoo", start, end)

st.subheader('Data from 2010-2022')
st.write(df.describe())

# Data visualization
st.subheader('Closing time vs time')
figure = plt.figure(figsize=(16, 6))
plt.plot(df.Close)
st.pyplot(figure)

# create dataset function

# feture extraction
df.shape
df = df['Close'].values
df = df.reshape(-1, 1)


dataset_test = np.array(df[-100:])


def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


# create testing dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_test = scaler.fit_transform(dataset_test)
x_test, y_test = create_dataset(dataset_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Load Model
model = load_model('stock_prediction.h5')

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

st.subheader('Predictions vs original')
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_facecolor('#000041')
ax.plot(y_test_scaled, color='red', label='Original price')
plt.plot(predictions, color='cyan', label='Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)
