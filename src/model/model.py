import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    # Input later (4 features of the flower) -->
    # Hidden layer 1 (Number of nodes) -->
    # H2 (n) -->
    # output (3 classes of iris flowers)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__() # instantiate our nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        # Push x into first layer
        x = F.relu(self.fc1(x))

        # Push x into second layer
        x = F.relu(self.fc2(x))

        # Set & return x as the output
        x = self.out(x)
        return x