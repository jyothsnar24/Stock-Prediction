
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import datetime
# from datetime import timedelta
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import streamlit as st
# import pytz  # Import the pytz library to handle timezones
# import pandas_datareader as pdr
# from pandas_datareader import data as pdr



# def get_stock_data(stock_name, num_data_points=5):
#     """
#     Fetch historical stock data from Yahoo Finance.

#     Parameters:
#         stock_name (str): The stock symbol of the company.
#         num_data_points (int): The number of data points to fetch.

#     Returns:
#         pd.DataFrame: A DataFrame containing the historical stock data.
#     """
#     try:
#         # Fetch the historical stock data from Yahoo Finance
#         stock_data = yf.download(stock_name, period='1mo')

#         # Select the relevant columns and limit the number of data points
#         stock_data = stock_data.tail(num_data_points)
#         stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
#         stock_data.reset_index(inplace=True)

#         return stock_data

#     except Exception as e:
#         print(f"Error fetching data for {stock_name}: {e}")
#         return None















# def predict_stock_price(symbol):
#     """
#     Predict the stock price for the next 7 days using the SARIMAX model.

#     Parameters:
#         symbol (str): The stock symbol of the company.

#     Returns:
#         pd.DataFrame: A DataFrame containing the predicted stock prices.
#     """
#     predictions_sarimax = pd.DataFrame()  # Empty DataFrame to store predictions
#     predicted_dates_sarima = pd.Index([])  # Empty Index to store predicted dates
#     predicted_prices_sarima = np.array([])  # Empty array to store predicted prices

#     # Set the end and start times for data retrieval
#     now = datetime.date.today()
#     end = now
#     start = end - timedelta(days=365)

#     # Retrieve stock data for the specified symbol
#     df = pdr.DataReader(symbol, data_source='yahoo', start=start, end=end)

#     # Create a new dataframe with only the 'Close' column
#     data = df.filter(['Close'])

#     # Convert the dataframe to a numpy array
#     dataset = data.values

#     # Get the number of rows to train the model on
#     training_data_len = int(np.ceil(len(dataset) * 0.95))

#     # Split the data into training and testing sets
#     train_data = dataset[0:training_data_len]
#     test_data = dataset[training_data_len:]

#     # Create the SARIMA model and fit it to the training data
#     sarimax_model = SARIMAX(train_data, order=(1, 0, 1), seasonal_order=(1, 1, 0, 12))
#     sarimax_model_fit = sarimax_model.fit()

#     # Make predictions for the next 7 days
#     predictions = sarimax_model_fit.forecast(steps=7)
#     predicted_prices_sarima = predictions

#     # Get the predicted dates for the next 7 days
#     last_date = df.index[-1]
#     predicted_dates_sarima = pd.Index(pd.date_range(start=last_date + pd.DateOffset(days=1), periods=7))

#     # Store the predicted dates and prices for the next 7 days in a dataframe
#     predictions_sarimax = pd.DataFrame({'Date': predicted_dates_sarima, 'SARIMAX Predictions': predicted_prices_sarima})

#     # Print the predicted dates and prices for the next 7 days
#     return predictions_sarimax









# def main():
#     st.title("Yahoo Finance Stock Prediction")

#     sample_stock_names = ["AAPL", "MSFT", "GOOGL","AMZN", "TSLA", "BRK.A", "META", "NVDA", "JPM", "BAC", "WFC", "V", "UNH", "JNJ", "MCD", "VZ", "PG", "COST", "KO", "HD", "MA", "VRTX", "WMT", "CRM", "INTC", "PEP", "TMO", "MRK", "ABBV", "MS", "ADBE", "LMT", "UNP", "CAT", "TXN", "CVX", "XOM", "DHR", "RTX", "IBM", "DOW", "NEE", "BA", "PFE", "HON", "CSCO", "LIN", "COSTCO", "MMM", "MDLZ", "TJX", "TGT", "NKE", "UPS", "TEL", "AVGO", "LLY", "CMG", "CI", "ORCL", "CHTR", "LOW", "ABT", "DHR", "DXCM", "ADP", "FISV", "NSC", "WM", "BK", "AEP", "EQT", "LHX", "ESRT", "WELL", "BKNG", "TPR", "EXPD", "IQV", "TDY", "DHR", "WBA", "LUMN", "DIS", "WM", "EFX", "KHC", "KLAC", "TROW", "MTCH", "SYK", "STZ", "ABMD", "DHR", "WDAY", "INTC", "WBA", "TRV", "DHR", "TJX", "XEL", "BBY", "TFC", "KLAC", "MS", "SBUX", "SYY", "DHR", "TJX", "XEL", "BBY", "TFC", "KLAC", "MS", "SBUX", "SYY"]

#     stock_name = st.selectbox("Select a Stock", sample_stock_names)

#     stock_data = get_stock_data(stock_name)

#     if stock_data is not None:
#         st.subheader(f"Stock Data for {stock_name}")
#         st.dataframe(stock_data)

#         if st.button("Predict"):
#             # Add a loading state while predicting
#             with st.spinner("Predicting..."):
#                 prediction_data = predict_stock_price(stock_name)

#             # Display the predicted data
#             if prediction_data is not None:
#                 # Use iloc to retrieve the 'Close' column
#                 prediction_data['Date'] = prediction_data['Date'].dt.date
#                 prediction_prices = prediction_data.iloc[:, 1]  # Extract the 'SARIMAX Predictions' column
#                 st.subheader(f"Future Prediction Prices for {stock_name}")
#                 st.dataframe(pd.DataFrame({"Date": prediction_data["Date"], "SARIMAX Predictions": prediction_prices}))
#             else:
#                 st.warning("No prediction data available.")
#     else:
#         st.error(f"Error fetching data for {stock_name}. Please try again later.")

# if __name__ == "__main__":
#     yf.pdr_override()
#     main()
    










import pandas_datareader as pdr
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def get_stock_data(stock_name, num_data_points=5):
    try:
        stock_data = yf.download(stock_name, period='1mo')
        stock_data = stock_data.tail(num_data_points)
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        stock_data.reset_index(inplace=True)
        return stock_data

    except Exception as e:
        st.error(f"Error fetching data for {stock_name}: {e}")
        return None

def predict_stock_price(symbol):
    try:
        predictions_sarimax = pd.DataFrame()
        predicted_dates_sarima = pd.Index([])
        predicted_prices_sarima = np.array([])

        now = datetime.now()
        end = now
        start = end - timedelta(days=365)

        df = pdr.DataReader(symbol, data_source='yahoo', start=start, end=end)
        data = df.filter(['Close'])
        dataset = data.values
        training_data_len = int(np.ceil(len(dataset) * 0.95))
        train_data = dataset[0:training_data_len]
        test_data = dataset[training_data_len:]

        sarimax_model = SARIMAX(train_data, order=(1, 0, 1), seasonal_order=(1, 1, 0, 12))
        sarimax_model_fit = sarimax_model.fit()
        
        predictions = sarimax_model_fit.forecast(steps=7)
        predicted_prices_sarima = predictions
        last_date = df.index[-1]
        predicted_dates_sarima = pd.Index(pd.date_range(start=last_date + pd.DateOffset(days=1), periods=7))
        predictions_sarimax = pd.DataFrame({'Date': predicted_dates_sarima, 'SARIMAX Predictions': predicted_prices_sarima})

        return predictions_sarimax

    except Exception as e:
        st.error(f"Error predicting stock price for {symbol}: {e}")
        return None

def main():
    st.title("Yahoo Finance Stock Prediction")
    sample_stock_names = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BRK.A"]
    stock_name = st.selectbox("Select a Stock", sample_stock_names)

    stock_data = get_stock_data(stock_name)

    if stock_data is not None:
        st.subheader(f"Stock Data for {stock_name}")
        st.dataframe(stock_data)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                prediction_data = predict_stock_price(stock_name)

            if prediction_data is not None:
                prediction_data['Date'] = prediction_data['Date'].dt.date
                prediction_prices = prediction_data.iloc[:, 1]
                st.subheader(f"Future Prediction Prices for {stock_name}")
                st.dataframe(pd.DataFrame({"Date": prediction_data["Date"], "SARIMAX Predictions": prediction_prices}))
            else:
                st.warning("No prediction data available.")
    else:
        st.error(f"Error fetching data for {stock_name}. Please try again later.")

if __name__ == "__main__":
    yf.pdr_override()
    main()

