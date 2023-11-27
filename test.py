# %%
import torch
import numpy as np
import twophase.coord as coord
import twophase.cubie as cubie
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

def generate_data(num_samples):
    # Initialize lists to store the values
    co_values, eo_values, ud_slice_values, N_values = [], [], [], []

    for _ in range(num_samples):
        cube = cubie.CubieCube()
        cube.randomize()

        # Get values for co, eo, and ud_slice
        co, eo, ud_slice = cube.get_twist(), cube.get_flip(), cube.get_slice()
        co_values.append(co)
        eo_values.append(eo)
        ud_slice_values.append(ud_slice)

        # Get the depth for phase 1 (N value)
        N = coord.CoordCube(cube).get_depth_phase1()
        N_values.append(N)

    # Convert lists to numpy arrays and then to PyTorch tensors
    X = np.column_stack([co_values, eo_values, ud_slice_values])
    N = np.array(N_values)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(N, dtype=torch.float32).view(-1, 1)

# Generate training and test datasets
train_X, train_Y = generate_data(50000)  # 8000 training samples
test_X, test_Y = generate_data(5000)     # 2000 test samples

# %%


# Create TensorDatasets for training and testing
train_dataset = TensorDataset(train_X, train_Y)
test_dataset = TensorDataset(test_X, test_Y)

# Create DataLoaders for training and testing
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64)


# %%
# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 40),  # Input layer with 3 inputs (x, y, z)
            nn.ReLU(),         # Activation function
            nn.Linear(40, 100),
            nn.ReLU(),         # Activation function
            nn.Linear(100, 100),
            nn.ReLU(),         # Activation function
            nn.Linear(100, 40),
            nn.ReLU(),         # Activation function
            nn.Linear(40, 1)   # Output layer with 1 output (n)
        )

    def forward(self, x):
        return self.layers(x)

# Initialize the model
model = MLP()

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Set matplotlib to interactive mode
plt.ion()

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 6))
train_losses = []

# Training loop
num_epochs = 60  # Adjust this as needed
for epoch in range(num_epochs):
    total_loss = 0.0
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Update the plot
    ax.clear()
    ax.plot(train_losses, label='Training Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss over Epochs')
    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Optional: print epoch number and loss
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

# Turn off interactive mode
plt.ioff()
plt.show()

# %%
model.eval()  # Set the model to evaluation mode
total_loss = 0.0

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

average_loss = total_loss / len(test_loader)
print(f'Average Test Loss: {average_loss:.4f}')


# %%
# pickle.dump(model, open("model_test.pickle", "wb"))


