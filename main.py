import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
df = pd.read_csv(url)

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

def train(model):
    # Train Test Split!
    # Set X to the flower features (features)
    # Set y to variety column (outcome)
    X = df.drop('variety', axis=1)
    y = df['variety']

    # Convert Data
    y = y.replace('Setosa', 0.0)
    y = y.replace('Versicolor', 1.0)
    y = y.replace('Virginica', 2.0)

    # OR convert to tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.long)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # Convert X features to float tensors
    X_train = torch.FloatTensor(X_train);
    X_test = torch.FloatTensor(X_test);

    # Convert y labels to tensors long
    y_train = torch.LongTensor(y_train);
    y_test = torch.LongTensor(y_test);

    # Set the criterion of model to measure the error, how far off the predictions are from
    criterion = nn.CrossEntropyLoss()

    # Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations(epochs)
    # we probably would want to lower our learning rate.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    # Epochs? (Process from start to output with all the training data)
    cycles = 200
    losses = []
    for i in range(cycles):
        # Go forward and create a prediction
        y_pred = model.forward(X_train) # Get predicted results

        # Measure the loss/error, high in the beginning
        loss = criterion(y_pred, y_train) # Predicted value vs y_train

        # Keep track of our losses
        losses.append(loss.item())

        # print every 10 cycles
        if i % 10 == 0:
            print(f'Cycle {i}/{cycles}, Loss: {loss.item()}')

        # Do some back propagation: take the error rate of foward propagation
        # and feed it back thru the network to fine tune the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(range(cycles), losses)
    plt.ylabel('Loss')
    plt.xlabel('Cycles')
    plt.show()

    ####


def main():
    # Set the random seed for reproducibility
    torch.manual_seed(41)

    # Create an instance of the model
    model = Model()

    # Train it
    train(model=model)

main()