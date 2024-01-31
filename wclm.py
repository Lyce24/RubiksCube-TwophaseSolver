import torch
import numpy as np
import twophase.coord as coord
import twophase.cubie as cubie
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def generate_data(num_samples):
    # Initialize lists to store the values
    X_values, N_values = [], []

    for _ in range(num_samples):
        cube = cubie.CubieCube()
        cube.randomize()
        co, eo, ud_slice = cube.get_twist(), cube.get_flip(), cube.get_slice()

        # easier => multiply matrix by vector
        co_oh = nn.functional.one_hot(torch.tensor([co]), 2187)
        eo_oh = nn.functional.one_hot(torch.tensor([eo]), 2048)
        ud_oh = nn.functional.one_hot(torch.tensor([ud_slice]), 495)

        inp = torch.cat((co_oh, eo_oh, ud_oh), dim=-1)

        X_values.append(inp[0])

        # Get the depth for phase 1 (N value)
        N = coord.CoordCube(cube).get_depth_phase1()
        N_values.append(N)

    # Convert lists to numpy arrays and then to PyTorch tensors
    X = np.array(X_values)
    N = np.array(N_values)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(N, dtype=torch.long).view(-1, 1)

train_X, train_Y = generate_data(50000)
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
print(class_weights)

# Create multi-class cross entropy loss model
class MCM(nn.Module):
    def __init__(self, num_classes):
        super(MCM, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4730 , 1024),   # Input layer with 3 inputs (x, y, z)
            nn.ReLU(),          # Activation function
            nn.Linear(1024, 512), # First hidden layer
            nn.ReLU(),          # Activation function
            nn.Linear(512, 256), # Second hidden layer
            nn.ReLU(),          # Activation function
            nn.Linear(256, 64),   # Output layer with 1 output (n)
            nn.ReLU(),          # Activation function
            nn.Linear(64, num_classes)    # Output layer with 1 output (n)
        )
    def forward(self, x):
        return self.layers(x)

net = MCM(max_label.item() + 1)  # Adding 1 because class count starts from 0
criterion = nn.CrossEntropyLoss(weight=class_weights.float())
optimizer = optim.Adam(net.parameters(), lr=0.001)

features = features.float()


prev_loss = 0
count = 0
for epoch in range(200):  # Adjust the number of epochs as necessary
    if count == 15:
        print("The loss has not changed in 15 epochs. Stopping training.")
        break
    optimizer.zero_grad()
    outputs = net(features)
    loss = criterion(outputs, labels.view(-1))  # Ensure labels are correctly shaped
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

net.eval()

for _ in range(100):
    cube = cubie.CubieCube()
    cube.randomize()
    co, eo, ud_slice = cube.get_twist(), cube.get_flip(), cube.get_slice()

    # easier => multiply matrix by vector
    co_oh = nn.functional.one_hot(torch.tensor([co]), 2187)
    eo_oh = nn.functional.one_hot(torch.tensor([eo]), 2048)
    ud_oh = nn.functional.one_hot(torch.tensor([ud_slice]), 495)

    X = torch.cat((co_oh, eo_oh, ud_oh), dim=-1)[0]
    N = coord.CoordCube(cube).get_depth_phase1()
    
    input = torch.tensor(X, dtype=torch.float32)
    predicted_outputs = net(input)
    predicted_label = torch.argmax(predicted_outputs, dim=-1)
    
    print(f'Actual N: {N}, Predicted N: {predicted_label.item()}')