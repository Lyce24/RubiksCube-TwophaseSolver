import torch
import numpy as np
import twophase.coord as coord
import twophase.cubie as cubie
import torch.nn as nn
import torch.optim as optim
import pickle
import math

cube = cubie.CubieCube()
cube.randomize()
co, eo, ud_slice = cube.get_twist(), cube.get_flip(), cube.get_slice()


# easier => multiply matrix by vector
a = nn.functional.one_hot(torch.tensor([co]), 2187)
b = nn.functional.one_hot(torch.tensor([eo]), 2048)
c = nn.functional.one_hot(torch.tensor([ud_slice]), 495)

inp = torch.cat((a, b, c), dim=-1)
# shape = (1, 4730), get the inner vector
inp = inp[0]

print(inp)