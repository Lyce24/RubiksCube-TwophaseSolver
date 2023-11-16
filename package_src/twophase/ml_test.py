import torch
import numpy as np
import twophase.coord as coord
import twophase.cubie as cubie
import torch.nn as nn
import torch.optim as optim
import pickle

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

filename = 'my_model_1.pickle'
loaded_model = pickle.load(open(filename, 'rb'))

sum = 0
for i in range(0, 1000):
    cube = cubie.CubieCube()
    cube.randomize()
    co, eo, ud_slice = cube.get_twist(), cube.get_flip(), cube.get_slice()
    
    input_data = torch.tensor([[co, eo, ud_slice]], dtype=torch.float32)
    with torch.no_grad():
        prediction = loaded_model(input_data)
        
    sum += prediction.item() - coord.CoordCube(cube).get_depth_phase1()

avg_loss = sum / 1000
print(f'Average Loss: {avg_loss:.4f}')
