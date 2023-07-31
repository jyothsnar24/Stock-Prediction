import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import timedelta
from fbprophet import Prophet


def get_stock_data(stock_name, num_data_points=5):
    """
    Fetch historical stock data from Yahoo Finance.

    Parameters:
        stock_name (str): The stock symbol of the company.
        num_data_points (int): The number of data points to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data.
    """
    try:
        # Fetch the historical stock data from Yahoo Finance
        stock_data = yf.download(stock_name, period='1mo')

        # Select the relevant columns and limit the number of data points
        stock_data = stock_data.tail(num_data_points)
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        stock_data.reset_index(inplace=True)

        return stock_data

    except Exception as e:
        print(f"Error fetching data for {stock_name}: {e}")
        return None





def predict_stock_price(symbol):
    """
    Predict the stock price for the next 7 days using the Prophet library.

    Parameters:
        symbol (str): The stock symbol of the company.

    Returns:
        pd.DataFrame: A DataFrame containing the predicted stock prices.
    """
    predictions_prophet = pd.DataFrame()  # Empty DataFrame to store predictions

    # Retrieve stock data for the specified symbol
    df = yf.download(symbol, period='1y', group_by='ticker')

    # Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])

    # Rename the columns to fit Prophet's requirements
    data = data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

    # Create the Prophet model and fit it to the data
    model = Prophet(daily_seasonality=True)
    model.fit(data)

    # Make predictions for the next 7 days
    future_dates = model.make_future_dataframe(periods=7)
    predictions = model.predict(future_dates)

    # Get the predicted dates and prices for the next 7 days
    predicted_dates = predictions.tail(7)['ds'].dt.date
    predicted_prices = predictions.tail(7)['yhat']

    # Store the predicted dates and prices for the next 7 days in a dataframe
    predictions_prophet = pd.DataFrame({'Date': predicted_dates, 'Prophet Predictions': predicted_prices})

    # Print the predicted dates and prices for the next 7 days
    return predictions_prophet




# def main():
#     st.title("Yahoo Finance Stock Prediction")
    
#     # Define a list of sample stock names
#     sample_stock_names = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
#     # Create a dropdown to select the stock name
#     stock_name = st.selectbox("Select a Stock", sample_stock_names)
    
#     # Get the historical stock data
#     stock_data = get_stock_data(stock_name)
    
#     if stock_data is not None:
#         st.subheader(f"Stock Data for {stock_name}")
#         st.dataframe(stock_data)

#         # Add a "Predict" button
#         if st.button("Predict"):
#             # Get predicted data using the predict_stock_price function
#             prediction_data = predict_stock_price(stock_name)

#             # Display the predicted data
#             if prediction_data is not None:
#                 st.subheader(f"Future Prediction Prices for {stock_name}")
#                 st.dataframe(prediction_data)
#             else:
#                 st.warning("No prediction data available.")
#     else:
#         st.error(f"Error fetching data for {stock_name}. Please try again later.")

# if __name__ == "__main__":
#     main()

def main():
    st.title("Yahoo Finance Stock Prediction")

    sample_stock_names = ["AAPL", "AMZN", "GOOGL", "MSFT", "FB", "TSLA", "BRK.A", "JPM", "BAC", "C", "XOM", "CVX", "GE", "NFLX", "DIS", "NVDA", "PYPL", "V", "MA", "JNJ", "PFE", "WMT", "KO", "NKE", "ADBE", "CRM", "MCD", "PG", "VZ", "IBM"]

    stock_name = st.selectbox("Select a Stock", sample_stock_names)

    stock_data = get_stock_data(stock_name)

    if stock_data is not None:
        st.subheader(f"Stock Data for {stock_name}")
        st.dataframe(stock_data)

        if st.button("Predict"):
            # Add a loading state while predicting
            with st.spinner("Predicting..."):
                prediction_data = predict_stock_price(stock_name)

            # Display the predicted data
            if prediction_data is not None:
                # Use iloc to retrieve the 'Close' column
                prediction_data['Date'] = prediction_data['Date'].dt.date
                prediction_prices = prediction_data.iloc[:, 1]  # Extract the 'SARIMAX Predictions' column
                st.subheader(f"Future Prediction Prices for {stock_name}")
                st.dataframe(pd.DataFrame({"Date": prediction_data["Date"], "SARIMAX Predictions": prediction_prices}))
            else:
                st.warning("No prediction data available.")
    else:
        st.error(f"Error fetching data for {stock_name}. Please try again later.")

if __name__ == "__main__":
    yf.pdr_override()
    main()
    
