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
    X_values, N_values = [], []

    for _ in range(num_samples):
        cube = cubie.CubieCube()
        cube.randomize()

        # Get values for co, eo, and ud_slice
        co, eo, ud_slice = cube.get_twist(), cube.get_flip(), cube.get_slice()
        co_oh = nn.functional.one_hot(torch.tensor([co]), 2187)
        eo_oh = nn.functional.one_hot(torch.tensor([eo]), 2048)
        ud_slice_oh = nn.functional.one_hot(torch.tensor([ud_slice]), 495)
        
        inp = torch.cat((co_oh, eo_oh, ud_slice_oh), dim=-1)[0]
        
        X_values.append(inp)
    

        # Get the depth for phase 1 (N value)
        N = coord.CoordCube(cube).get_depth_phase1()
        N_values.append(N)

    # Convert lists to numpy arrays and then to PyTorch tensors
    X = np.array(X_values)
    N = np.array(N_values)

    return torch.tensor(X, dtype=torch.float), torch.tensor(N, dtype=torch.long).view(-1, 1)


# %%
# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4730, 1024),   # Input layer with 3 inputs (x, y, z)
            nn.ReLU(),          # Activation function
            nn.Linear(1024, 256), # First hidden layer
            nn.ReLU(),          # Activation function
            nn.Linear(256, 13)  
        )

    def forward(self, x):
        return self.layers(x)

# Initialize the model
model = MLP()  # Adding 1 because class count starts from 0

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Set matplotlib to interactive mode
plt.ion()

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 6))
train_losses = []

# Training loop
num_epochs = 5000  # Adjust this as needed
prev_loss = 0
count = 0
for epoch in range(num_epochs):
    train_X, train_Y = generate_data(20000)  # 50000 training sample
    
    features = train_X
    labels = train_Y

    # construct a dictionary to store the index of each label
    move_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 
                 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 
                 12:0}

    # find the index of each label
    for i in range(labels.shape[0]):
        move_dict[labels[i].item()] += 1
    
    class_count = []
    for key in move_dict:
        class_count.append(move_dict[key])

    class_counts = torch.tensor(class_count)
    total_samples = labels.shape[0]
    class_weights = total_samples / (class_counts * len(class_counts))
    criterion = nn.CrossEntropyLoss(weight=class_weights.float())
        # Create TensorDataset
    train_data = TensorDataset(features, labels)

    # Define batch size
    batch_size = 64  # This is just an example, you might want to adjust this

    # Create DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    
    model.train()  # Set the model to training mode
    if count == 15:
        print("The loss has not changed in 15 epochs. Stopping training.")
        break
    else:         
        total_loss = 0.0           
    
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1))  # Ensure labels are correctly shaped

            # Backward and optimize

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if abs(prev_loss - avg_loss) < 0.005:
            count += 1
        else:
            
            count = 0
        prev_loss = avg_loss
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

# Turn off interactive mode and save the plot
plt.savefig('training_loss.png')
plt.ioff()
plt.show()


# %%
model.eval()  # Set the model to evaluation mode

i = 0
for _ in range(1000):
    cube = cubie.CubieCube()
    cube.randomize()
    co, eo, ud_slice = cube.get_twist(), cube.get_flip(), cube.get_slice()
    
    co_oh = nn.functional.one_hot(torch.tensor([co]), 2187)
    eo_oh = nn.functional.one_hot(torch.tensor([eo]), 2048)
    ud_slice_oh = nn.functional.one_hot(torch.tensor([ud_slice]), 495)
    
    X = torch.cat((co_oh, eo_oh, ud_slice_oh), dim=-1)[0]
    N = coord.CoordCube(cube).get_depth_phase1()
    
    # Get the predicted N value
    input = torch.tensor(X, dtype=torch.float32)
    predicted_output = model(input)
    predicted_label = torch.argmax(predicted_output, dim=-1)
    
    if predicted_label.item() == N:
        i += 1
    
    # Print the predicted and actual N values
    print(f'Predicted N: {predicted_label.item()}, Actual N: {N}')

print(f"Accuracy: {i/1000}")

# # %%
# pickle.dump(model, open("model_test.pickle", "wb"))


