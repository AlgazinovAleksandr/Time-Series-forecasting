from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_prophet_model(train, val):
    """
    Combines train and validation dataframes, trains a Prophet model using the 'OT' variable, and returns the model.
    """
    # Combine train and validation datasets
    combined_data = pd.concat([train, val])

    # Prepare the data for Prophet
    # Prophet requires columns named 'ds' for the date and 'y' for the target variable
    prophet_data = combined_data[['date', 'OT']].rename(columns={'date': 'ds', 'OT': 'y'})

    # Initialize and train the Prophet model
    model = Prophet()
    model.fit(prophet_data)

    return model


def evaluate_prophet_model(model, test, n_steps=None, save_plot=False, plot_filename="forecast_plot.png"):
    """
    Evaluates the Prophet model on the test dataframe by predicting n_steps ahead.
    Calculates MSE and MAE, plots predictions vs actual values, and optionally saves the plot.
    """
    # Set default n_steps to the length of the test dataframe if not provided
    if n_steps is None:
        n_steps = len(test)

    # Prepare the future dataframe for prediction
    future = test[['date']].rename(columns={'date': 'ds'}).iloc[:n_steps]

    # Make predictions
    forecast = model.predict(future)

    # Extract the predicted values and actual values
    y_pred = forecast['yhat'].values
    y_true = test['OT'].iloc[:n_steps].values

    # Calculate MSE and MAE
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Print the metrics
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(test['date'].iloc[:n_steps], y_true, label="Actual", color="blue")
    plt.plot(test['date'].iloc[:n_steps], y_pred, label="Predicted", color="orange")
    plt.xlabel("Date")
    plt.ylabel("OT")
    plt.title("Prophet Predictions vs Actual Values")
    plt.legend()
    plt.grid()

    # Save the plot if requested
    if save_plot:
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")

    # Show the plot
    plt.show()

    return mse, mae

def train_and_evaluate(train, val, test, n_steps=None, save_plot=False, plot_filename="forecast_plot.png"):
    """
    Combines training and evaluation of the Prophet model.
    Trains the model on train and validation data, evaluates it on the test data, and returns the metrics, as well as the model itself.
    """
    # Train the Prophet model
    model = train_prophet_model(train, val)

    # Evaluate the model on the test dataset
    mse, mae = evaluate_prophet_model(model, test, n_steps=n_steps, save_plot=save_plot, plot_filename=plot_filename)

    return model, mse, mae