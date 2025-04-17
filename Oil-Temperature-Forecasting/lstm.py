import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Custom Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, forecast_steps):
        self.data = data
        self.seq_length = seq_length
        self.forecast_steps = forecast_steps

    def __len__(self):
        return len(self.data) - self.seq_length - self.forecast_steps + 1

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx+self.seq_length]
        target = self.data[idx+self.seq_length:idx+self.seq_length+self.forecast_steps, -1] # Assuming last column is target
        return torch.FloatTensor(input_seq), torch.FloatTensor(target)

# LSTM Model
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    

def train_lstm(model, train_loader, val_loader, device, LEARNING_RATE, N_EPOCHS, N_EARLY_STOP, save_path, criterion=nn.MSELoss()):
    

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop with early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(N_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{N_EPOCHS}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= N_EARLY_STOP:
                print(f'Early stopping after {epoch+1} epochs')
                break

def evaluate_lstm(model, test_loader, n_steps=None, save_plot=False, plot_filename="forecast_plot.png"):
    model.eval()
    device = next(model.parameters()).device
    all_outputs = []
    all_targets = []
    
    # Collect predictions and targets
    with torch.no_grad():
        samples_collected = 0
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Convert to numpy arrays
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # Handle limited number of steps
            if n_steps is not None:
                remaining = n_steps - samples_collected
                if remaining <= 0:
                    break
                outputs_np = outputs_np[:remaining]
                targets_np = targets_np[:remaining]
            
            all_outputs.append(outputs_np)
            all_targets.append(targets_np)
            
            if n_steps is not None:
                samples_collected += len(outputs_np)
                if samples_collected >= n_steps:
                    break
    
    # Concatenate and truncate if necessary
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    if n_steps is not None:
        all_outputs = all_outputs[:n_steps]
        all_targets = all_targets[:n_steps]
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_outputs)
    mae = mean_absolute_error(all_targets, all_outputs)
    
    # Plotting all predictions
    plt.figure(figsize=(12, 6))
    plt.plot(all_targets.flatten(), label='Real Values')
    plt.plot(all_outputs.flatten(), label='Forecasted Values', alpha=0.7)
    plt.title(f'Real vs Forecasted OT Values ({len(all_targets)} Samples)')
    plt.xlabel('Time Steps')
    plt.ylabel('OT Value')
    plt.legend()

    # Save the plot if requested
    if save_plot:
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")

    plt.show()
    
    return mse, mae, all_outputs, all_targets


def evaluate_single_forecast(model, val_data, test_data, seq_length, forecast_steps, features, save_plot=False, plot_filename="forecast_plot.png"):
    # Get last sequence from validation data
    last_sequence = val_data[features][-seq_length:].values
    device = next(model.parameters()).device
    
    # Prepare input tensor
    input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy().flatten()
    
    test_actual = test_data['OT'][:forecast_steps].values
    
    # Calculate metrics
    mse = mean_squared_error(test_actual, prediction)
    mae = mean_absolute_error(test_actual, prediction)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    time_pred = np.arange(0, forecast_steps)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(time_pred, prediction, label='Forecast', linestyle='--')
    plt.plot(time_pred, test_actual, label='Actual Test Values', alpha=0.7)
    
    plt.title(f'{forecast_steps}-Step Forecast vs Actual')
    plt.xlabel('Time Steps (0 = Forecast Start)')
    plt.ylabel('OT Value')
    plt.legend()
    plt.grid(True)

    if save_plot:
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")

    plt.show()
    
    return mse, mae

