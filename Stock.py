import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

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
    Predict the stock price for the next 7 days using the SARIMAX model.

    Parameters:
        symbol (str): The stock symbol of the company.

    Returns:
        pd.DataFrame: A DataFrame containing the predicted stock prices.
    """

    # Set the end and start times for data retrieval
    end = datetime.now()
    start = end - timedelta(days=365)

    # Retrieve stock data for AAPL
    yf.pdr_override()
    df = pdr.get_data_yahoo(symbol, start, end)

    # Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])

    # Convert the dataframe to a numpy array
    dataset = data.values

    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * 0.95))

    # Split the data into training and testing sets
    train_data = dataset[0:training_data_len]
    test_data = dataset[training_data_len:]

    # Create the SARIMA model and fit it to the training data
    sarimax_model = SARIMAX(train_data, order=(1, 0, 1), seasonal_order=(1, 1, 0, 12))
    sarimax_model_fit = sarimax_model.fit()

    # Make predictions for the next 7 days
    predictions = sarimax_model_fit.forecast(steps=7)
    predicted_prices_sarima = predictions

    # Get the predicted dates for the next 7 days
    last_date = df.index[-1]
    predicted_dates_sarima = pd.Index(pd.date_range(start=last_date + pd.DateOffset(days=1), periods=7))

    # Store the predicted dates and prices for the next 7 days in a dataframe
    predictions_sarimax = pd.DataFrame({'Date': predicted_dates_sarima, 'SARIMAX Predictions': predicted_prices_sarima})

    # Print the predicted dates and prices for the next 7 days
    return predictions_sarimax



def main():
    st.title("Yahoo Finance Stock Prediction")
    
    # Define a list of sample stock names
    sample_stock_names = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    # Create a dropdown to select the stock name
    stock_name = st.selectbox("Select a Stock", sample_stock_names)
    
    # Get the historical stock data
    stock_data = get_stock_data(stock_name)
    
    if stock_data is not None:
        st.subheader(f"Stock Data for {stock_name}")
        st.dataframe(stock_data)

        # Add a "Predict" button
        if st.button("Predict"):
            # Get predicted data using the predict_stock_price function
            prediction_data = predict_stock_price(stock_name)

            # Display the predicted data
            if prediction_data is not None:
                st.subheader(f"Future Prediction Prices for {stock_name}")
                st.dataframe(prediction_data)
            else:
                st.warning("No prediction data available.")
    else:
        st.error(f"Error fetching data for {stock_name}. Please try again later.")

if __name__ == "__main__":
    main()
