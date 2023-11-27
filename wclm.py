import torch
import numpy as np
import twophase.coord as coord
import twophase.cubie as cubie
import torch.nn as nn
import torch.optim as optim

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

    return torch.tensor(X, dtype=torch.int64), torch.tensor(N, dtype=torch.long).view(-1, 1)

# Generate training and test datasets
train_X, train_Y = generate_data(5000000)
test_X, test_Y = generate_data(1000)

features = train_X
labels = train_Y

# find the max value in label
max_label = torch.max(labels)

# construct a dictionary to store the index of each label
move_dict = {}
for i in range(max_label+1):
    move_dict[i] = 0

# find the index of each label
for i in range(labels.shape[0]):
    move_dict[labels[i].item()] += 1
   
class_count = [] 
for key in move_dict:
    class_count.append(move_dict[key])

class_counts = torch.tensor(class_count)
total_samples = labels.shape[0]
class_weights = total_samples / (class_counts * len(class_counts))

print(move_dict)

# Create multi-class cross entropy loss model
class MCM(nn.Module):
    def __init__(self, num_classes):
        super(MCM, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),   # Input layer with 3 inputs (x, y, z)
            nn.ReLU(),          # Activation function
            nn.Linear(64, 128), # First hidden layer
            nn.ReLU(),          # Activation function
            nn.Linear(128, 64), # Second hidden layer
            nn.ReLU(),          # Activation function
            nn.Linear(64, num_classes)    # Output layer with 1 output (n)
        )
    def forward(self, x):
        return self.layers(x)

net = MCM(max_label.item() + 1)  # Adding 1 because class count starts from 0
criterion = nn.CrossEntropyLoss(weight=class_weights.float())
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Convert features to float for model compatibility
features = features.float()

# Training loop
for epoch in range(5000):  # Adjust the number of epochs as necessary
    optimizer.zero_grad()
    outputs = net(features)
    loss = criterion(outputs, labels.view(-1))  # Ensure labels are correctly shaped
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
net.eval()

for _ in range(100):
    co_values, eo_values, ud_slice_values, N_values = [], [], [], []
    cube = cubie.CubieCube()
    cube.randomize()

        # Get values for co, eo, and ud_slice
    co, eo, ud_slice = cube.get_twist(), cube.get_flip(), cube.get_slice()
    co_values.append(co)
    eo_values.append(eo)
    ud_slice_values.append(ud_slice)

    # Get the depth for phase 1 (N value)
    N = coord.CoordCube(cube).get_depth_phase1()
    X = np.column_stack([co_values, eo_values, ud_slice_values])
    
    input = torch.tensor(X, dtype=torch.float32)
    predicted_outputs = net(input)
    predicted_label = torch.argmax(predicted_outputs, dim=1)
    
    print(f'Actual N: {N}, Predicted N: {predicted_label.item()}')