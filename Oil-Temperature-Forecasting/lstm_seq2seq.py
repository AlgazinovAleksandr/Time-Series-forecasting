import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import random
import torch.optim as optim
import time # Optional: for timing epochs
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, forecast_steps):
        self.data = data
        self.seq_length = seq_length
        self.forecast_steps = forecast_steps
        # Assuming data is a numpy array [num_samples, num_features]
        # Assuming the target 'OT' is the last column (-1)
        self.num_features = data.shape[1]

    def __len__(self):
        # Ensure enough data for input sequence AND target sequence
        return len(self.data) - self.seq_length - self.forecast_steps + 1

    def __getitem__(self, idx):
        input_seq = self.data[idx : idx + self.seq_length]
        # Target sequence contains only the 'OT' feature for future steps
        target_seq = self.data[idx + self.seq_length : idx + self.seq_length + self.forecast_steps, -1]
        return torch.FloatTensor(input_seq), torch.FloatTensor(target_seq)
    


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # If num_layers > 1, dropout applies between LSTM layers (except last)
        # batch_first=True means input/output tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout) # Optional extra dropout on output

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # Initialize hidden and cell states (optional, LSTM defaults to zeros)

        # outputs contains output features for each time step
        # hidden/cell are the final states of the sequence
        outputs, (hidden, cell) = self.lstm(x) # , (h0, c0) -> optional initial state

        # Optional: Apply dropout to the final hidden/cell states if desired
        # hidden = self.dropout(hidden)
        # cell = self.dropout(cell)

        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.output_size = output_size # Should be 1 (predicting 'OT' only)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Decoder input size is 1 (previous 'OT' value)
        self.lstm = nn.LSTM(1, hidden_size, num_layers, # Input is single value
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        # x shape: (batch_size, 1) -> Need to unsqueeze for seq dimension
        # Unsqueeze to add sequence length dimension of 1
        x = x.unsqueeze(1) # shape: (batch_size, 1, 1)

        # Pass input and previous states to LSTM
        # output shape: (batch_size, 1, hidden_size)
        # hidden/cell shape: (num_layers, batch_size, hidden_size)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))

        # Apply dropout to LSTM output before linear layer
        output = self.dropout(output.squeeze(1)) # shape: (batch_size, hidden_size)

        # Pass through linear layer to get prediction
        prediction = self.fc(output) # shape: (batch_size, output_size) -> (batch_size, 1)

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        # Ensure output_size of decoder is accessible if needed later
        self.decoder_output_size = decoder.output_size

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src shape: (batch_size, src_len, input_size)
        # trg shape: (batch_size, trg_len) -> target sequence for teacher forcing
        # teacher_forcing_ratio: probability to use real target as next input

        batch_size = src.shape[0]
        trg_len = trg.shape[1] # This defines the forecast_steps

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, self.decoder_output_size).to(self.device)

        # Encode the source sequence to get context (final hidden/cell states)
        hidden, cell = self.encoder(src)

        # --- Decoder Loop ---
        # First input to the decoder:
        # Option 1: Use the last observed 'OT' value from the input sequence
        # Assumes 'OT' is the last feature in src
        decoder_input = src[:, -1, -1].unsqueeze(1) # shape: (batch_size, 1)
        # Option 2: Use a zero or start token (if applicable)
        # decoder_input = torch.zeros(batch_size, 1).to(self.device)

        for t in range(trg_len):
            # Decode one step
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # Store the prediction for this step
            # decoder_output shape is (batch_size, output_size=1)
            outputs[:, t, :] = decoder_output

            # Decide whether to use teacher forcing for the next input
            use_teacher_force = random.random() < teacher_forcing_ratio

            if use_teacher_force:
                # Use actual target value from this time step as next input
                # Unsqueeze trg[:, t] to make it (batch_size, 1)
                decoder_input = trg[:, t].unsqueeze(1)
            else:
                # Use the decoder's own prediction as the next input
                # Ensure it has the correct shape (batch_size, 1)
                decoder_input = decoder_output # Already (batch_size, 1)

        # outputs shape: (batch_size, trg_len, output_size=1)
        # Squeeze the last dimension if output_size is 1 for compatibility with loss
        if self.decoder_output_size == 1:
            outputs = outputs.squeeze(-1) # shape: (batch_size, trg_len)

        return outputs


def train_seq2seq(model, train_loader, val_loader, device, LEARNING_RATE, N_EPOCHS, N_EARLY_STOP, save_path, criterion=nn.MSELoss(), teacher_forcing_start_ratio=0.5):
    """Trains a Seq2Seq model."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    early_stop_counter = 0
    # Optional: Adjust teacher forcing ratio over epochs
    teacher_forcing_ratio = teacher_forcing_start_ratio

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs shape: (batch, seq_len, num_features)
            # targets shape: (batch, forecast_steps)

            optimizer.zero_grad()

            # Pass inputs and targets (for teacher forcing) to the model
            # Use the current teacher_forcing_ratio
            outputs = model(inputs, targets, teacher_forcing_ratio)
            # outputs shape: (batch, forecast_steps)

            loss = criterion(outputs, targets)
            loss.backward()
            # Optional: Gradient Clipping
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # **IMPORTANT**: Use teacher_forcing_ratio = 0.0 during validation/testing
                # The model must predict based on its own outputs
                outputs = model(inputs, targets, 0.0) # NO teacher forcing

                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # Calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        # Optional: Update learning rate scheduler
        scheduler.step(val_loss)

        # Optional: Decay teacher forcing ratio
        # Example: Linear decay
        # teacher_forcing_ratio = max(0.0, teacher_forcing_start_ratio - (epoch * (teacher_forcing_start_ratio / (N_EPOCHS * 0.75))))


        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{N_EPOCHS} | Time: {epoch_time:.2f}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        # print(f'\tTeacher Forcing Ratio: {teacher_forcing_ratio:.4f}') # If decaying

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f'\tValidation loss decreased ({best_val_loss:.4f}). Saving model...')
        else:
            early_stop_counter += 1
            print(f'\tValidation loss did not improve. Early stopping counter: {early_stop_counter}/{N_EARLY_STOP}')
            if early_stop_counter >= N_EARLY_STOP:
                print(f'--- Early stopping triggered after {epoch+1} epochs ---')
                break

    print(f'Training finished. Best validation loss: {best_val_loss:.4f}')
    # Load the best model state
    model.load_state_dict(torch.load(save_path))
    return model # Return the loaded best model

def evaluate_seq2seq(model, test_loader, criterion=nn.MSELoss(), n_steps=None, save_plot=False, plot_filename="forecast_plot_seq2seq.png"):
    """Evaluates the Seq2Seq model on the test set."""
    model.eval()
    device = next(model.parameters()).device
    all_outputs = []
    all_targets = []
    test_loss = 0.0

    with torch.no_grad():
        samples_collected = 0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs shape: (batch, seq_len, num_features)
            # targets shape: (batch, forecast_steps)

            # IMPORTANT: NO teacher forcing during evaluation
            outputs = model(inputs, targets, 0.0)
            # outputs shape: (batch, forecast_steps)

            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

            # Convert to numpy arrays
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Handle limited number of steps (samples)
            if n_steps is not None:
                remaining = n_steps - samples_collected
                if remaining <= 0:
                    break
                # Select subset of the batch if needed
                current_batch_size = outputs_np.shape[0]
                num_to_take = min(remaining, current_batch_size)
                outputs_np = outputs_np[:num_to_take]
                targets_np = targets_np[:num_to_take]

            all_outputs.append(outputs_np)
            all_targets.append(targets_np)

            if n_steps is not None:
                samples_collected += len(outputs_np)
                if samples_collected >= n_steps:
                    break

    # Concatenate results from all batches
    # Check if any predictions were made
    if not all_outputs:
        print("No samples processed in evaluation.")
        return None, None, None, None, None

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate overall loss
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f'Average Test Loss: {avg_test_loss:.4f}')

    # Calculate metrics (MSE, MAE)
    # Ensure predictions and targets are 2D [samples, steps] before flattening if needed
    # For direct comparison, compare each step:
    mse_per_step = mean_squared_error(all_targets, all_outputs, multioutput='raw_values')
    mae_per_step = mean_absolute_error(all_targets, all_outputs, multioutput='raw_values')
    mse_overall = mean_squared_error(all_targets, all_outputs)
    mae_overall = mean_absolute_error(all_targets, all_outputs)

    print(f"Overall Mean Squared Error (MSE): {mse_overall:.4f}")
    print(f"Overall Mean Absolute Error (MAE): {mae_overall:.4f}")
    print(f"MSE per forecast step: {mse_per_step}")
    print(f"MAE per forecast step: {mae_per_step}")


    # --- Plotting ---
    # Plotting flattened results (shows overall distribution but not individual forecasts well)
    plt.figure(figsize=(15, 6))
    plt.plot(all_targets.flatten(), label='Real Values (Flattened)')
    plt.plot(all_outputs.flatten(), label='Forecasted Values (Flattened)', alpha=0.7)
    plt.title(f'Overall Real vs Forecasted OT Values ({len(all_targets)} Samples x {all_targets.shape[1]} Steps)')
    plt.xlabel('Time Steps (Flattened)')
    plt.ylabel('OT Value')
    plt.legend()
    plt.grid(True)
    if save_plot:
        plt.savefig(plot_filename.replace('.png', '_flat.png'))
        print(f"Flattened plot saved as {plot_filename.replace('.png', '_flat.png')}")
    plt.show()

    # Plotting example forecasts (more insightful)
    n_examples_to_plot = min(5, len(all_outputs)) # Plot up to 5 examples
    plt.figure(figsize=(15, 4 * n_examples_to_plot))
    for i in range(n_examples_to_plot):
        plt.subplot(n_examples_to_plot, 1, i + 1)
        plt.plot(all_targets[i], label='Real Values', marker='o')
        plt.plot(all_outputs[i], label='Forecasted Values', linestyle='--', marker='x')
        plt.title(f'Example Forecast {i+1}')
        plt.xlabel('Forecast Step')
        plt.ylabel('OT Value')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    if save_plot:
        plt.savefig(plot_filename.replace('.png', '_examples.png'))
        print(f"Example forecasts plot saved as {plot_filename.replace('.png', '_examples.png')}")
    plt.show()


    return mse_overall, mae_overall, all_outputs, all_targets, avg_test_loss


def evaluate_single_seq2seq_forecast(model, val_data, test_data, seq_length, forecast_steps, features, target_col_index=-1, save_plot=False, plot_filename="single_forecast_plot_seq2seq.png"):
    """Evaluates a single forecast starting from the end of validation data."""
    model.eval()
    device = next(model.parameters()).device

    # Get last sequence from validation data (all features)
    last_sequence = val_data[features].iloc[-seq_length:].values # Use .iloc for positional slicing if DataFrame

    # Prepare input tensor
    input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device) # Add batch dimension

    # Get the actual target values from the beginning of the test data
    test_actual = test_data['OT'].iloc[:forecast_steps].values # Assuming 'OT' is the target column name

    # Create a dummy target tensor for the model call (needed for structure, but not used with ratio=0)
    # Shape: (batch_size=1, forecast_steps)
    dummy_target = torch.FloatTensor(test_actual).unsqueeze(0).to(device)

    # Make prediction - NO teacher forcing
    with torch.no_grad():
        prediction_tensor = model(input_tensor, dummy_target, 0.0) # teacher_forcing_ratio = 0.0
        prediction = prediction_tensor.cpu().numpy().flatten() # Shape (forecast_steps,)

    # Ensure prediction and actual have the same length (handle edge cases if needed)
    min_len = min(len(prediction), len(test_actual))
    prediction = prediction[:min_len]
    test_actual = test_actual[:min_len]

    # Calculate metrics for this single forecast
    mse = mean_squared_error(test_actual, prediction)
    mae = mean_absolute_error(test_actual, prediction)

    print(f"Single Forecast Evaluation ({forecast_steps} steps):")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")

    time_pred = np.arange(min_len)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(time_pred, prediction, label='Forecast', linestyle='--', marker='x')
    plt.plot(time_pred, test_actual, label='Actual Test Values', alpha=0.7, marker='o')
    plt.title(f'Single {forecast_steps}-Step Forecast vs Actual (Starting after Validation Data)')
    plt.xlabel('Time Steps into Forecast Horizon')
    plt.ylabel('OT Value')
    plt.legend()
    plt.grid(True)

    if save_plot:
        plt.savefig(plot_filename)
        print(f"Single forecast plot saved as {plot_filename}")

    plt.show()

    return mse, mae, prediction, test_actual