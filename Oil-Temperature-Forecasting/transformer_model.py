import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE
#from pytorch_lightning import Trainer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html - reference

# Create a time index from the date. In this example, we set the minimum date of the training set as time zero.

def prepare_data(train_df, val_df, test_df, max_encoder_length=96, max_prediction_length=96, batch_size=128):
    """
    max_encoder_length - historical window length
    max_prediction_length - number of forecast steps (n-forecast_steps)
    """
    train_df = train_df.sort_values("date")
    min_date = train_df["date"].min()
    train_df["time_idx"] = ((train_df["date"] - min_date).dt.total_seconds() // 3600).astype(int)
    val_df["time_idx"] = ((val_df["date"] - min_date).dt.total_seconds() // 3600).astype(int)
    test_df["time_idx"] = ((test_df["date"] - min_date).dt.total_seconds() // 3600).astype(int)

    # For PyTorch Forecasting, you need a group id for each time series.
    # If you only have one time series, you can add a constant group id.
    train_df["series_id"] = 0
    val_df["series_id"] = 0
    test_df["series_id"] = 0

#     target_normalizer = GroupNormalizer(
#     groups=["series_id"], 
#     method="standard"  # Applies standard scaling (subtract mean, divide by std)
# )

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="OT",
        group_ids=["series_id"],
        min_encoder_length=max_encoder_length,  # allow flexibility if needed
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        #target_normalizer=target_normalizer,
        time_varying_unknown_reals = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        add_relative_time_idx=True,  # helpful for positional encodings internally
        #add_target_scales=True,      # provides target scaling information to the model
        add_encoder_length=True      # optionally include encoder length as a feature
    )

    # Create a validation dataset from the training dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, 
        val_df, 
        predict=True,        # tells the dataset to create a prediction dataset
        stop_randomization=True
    )

    test_dataset = TimeSeriesDataSet.from_dataset(
    training,      # This passes the configuration along with the computed normalization parameters
    test_df, 
    predict=True,  # Indicates that this dataset is for prediction rather than training
    stop_randomization=True  # Ensures deterministic behavior for evaluation
)
    
    # Batch sizes can be tuned. Here batch_size=64 is used as an example.
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader   = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    return training, validation, test_dataset, train_dataloader, val_dataloader, test_dataloader

# ===============================
# 4. Define and Train the Model
# ===============================

def train_transformer(training, train_dataloader, val_dataloader, save_file_name, max_prediction_length=96, hidden_size=16, hidden_continuous_size=8, attention_head_size=1, dropout=0.1):
    # Set up a PyTorch Lightning Trainer. Use "gpus=1" if you have a GPU available.
    pl.seed_everything(42)
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=True,
        mode="min"
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename=save_file_name,
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        # clipping gradients is a hyperparameter and important to prevent divergance
        # of the gradient for recurrent neural networks
        max_epochs=100,  # Hard limit of 20 epochs
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],  # Add both callbacks
        enable_progress_bar=True,
    )

    # Create the Temporal Fusion Transformer model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=hidden_size,         # size of hidden layers in the network
        attention_head_size=attention_head_size,  # number of attention heads
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=1,          # output size: can be adjusted based on whether you output quantiles, etc.
        loss=MAE(),           # using MAE as an example loss
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    # Train the model
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    return tft

def evaluate_transformer_model(y_pred, test, n_steps=96, save_plot=False, plot_filename="forecast_plot.png"):
    """
    Evaluates the Prophet model on the test dataframe by predicting n_steps ahead.
    Calculates MSE and MAE, plots predictions vs actual values, and optionally saves the plot.
    """
    # Set default n_steps to the length of the test dataframe if not provided
    

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
    plt.title("Transformer Predictions vs Actual Values")
    plt.legend()
    plt.grid()

    # Save the plot if requested
    if save_plot:
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")

    # Show the plot
    plt.show()

    return mse, mae


def forecast_and_save_results(train, val, test, name_df, max_encoder_length, max_prediction_length, save_file_name, plot_filename, metrics):

    if max_prediction_length == 96:
        hidden_size = 16
        hidden_continuous_size = 8
        attention_head_size = 1
        dropout = 0.1
    elif max_prediction_length == 192:
        hidden_size = 32
        hidden_continuous_size = 16
        attention_head_size = 2
        dropout = 0.3

    training, validation, test_set, train_dataloader, val_dataloader, test_dataloader = prepare_data(train, val, test, max_encoder_length=max_encoder_length, max_prediction_length=max_prediction_length, batch_size=256)
    transformer_model = train_transformer(training, train_dataloader, val_dataloader, save_file_name=save_file_name, max_prediction_length=max_prediction_length, hidden_size=hidden_size, hidden_continuous_size=hidden_continuous_size, attention_head_size=attention_head_size, dropout=dropout)

    predictions = transformer_model.predict(test_dataloader)
    predictions = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    predictions = predictions[0]

    mse_transformer, mae_transformer = evaluate_transformer_model(predictions, test, n_steps=max_prediction_length, save_plot=True, plot_filename=plot_filename)

    new_row = pd.DataFrame([['transformer', max_prediction_length, name_df, mse_transformer, mae_transformer]],
                        columns=metrics.columns)
    metrics = pd.concat([metrics, new_row], ignore_index=True)
    return metrics