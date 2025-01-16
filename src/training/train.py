import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn

def prepare_data(df):
    """
    Prepares the data by converting to tensors and splitting into train/test sets.

    Args:
        df (pd.DataFrame): The dataset with features and target.

    Returns:
        tuple: X_train, X_test, y_train, y_test (all torch tensors)
    """
    # Set X to the features and y to the variety (target column)
    X = df.drop('variety', axis=1)
    y = df['variety']

    # Map string labels to numeric values
    label_map = {'Setosa': 0.0, 'Versicolor': 1.0, 'Virginica': 2.0}
    y = y.replace(label_map)

    # Convert to PyTorch tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.long)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=41)

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train, learning_rate=0.01, cycles=100):
    """
    Trains the given PyTorch model on the training data.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        X_train (torch.Tensor): Training feature data.
        y_train (torch.Tensor): Training target labels.
        learning_rate (float): Learning rate for the optimizer.
        cycles (int): Number of training cycles (epochs).

    Returns:
        list: A list of loss values recorded during training.
    """
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    for i in range(cycles):
        # Forward pass: Predict the labels
        y_pred = model(X_train)

        # Compute loss
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())

        # Print loss every 10 cycles
        if i % 10 == 0:
            print(f'Cycle {i}/{cycles}, Loss: {loss.item()}')

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses


def plot_loss(losses):
    """
    Plots the loss over training cycles.

    Args:
        losses (list): A list of loss values recorded during training.
    """
    plt.plot(range(len(losses)), losses)
    plt.ylabel('Loss')
    plt.xlabel('Cycles')
    plt.title('Training Loss')
    plt.show()

def evaluate_test_data(model, X_test, y_test):
    correct = 0
    with torch.no_grad():  # Basically turn off back propagation.
        for i, data in enumerate(X_test, 0):
            y_val = model.forward(data)
            print(f'{i + 1}.) {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

            # Correct or not
            if y_val.argmax().item() == y_test[i]:
                correct += 1

    print(f'{correct} correct')

def train(model, df):
    """
    Main training function to prepare data, train the model, and visualize results.

    Args:
        df (pd.DataFrame): Dataset with features and target.
        model (torch.nn.Module): PyTorch model to train.
    """
    # Step 1: Prepare the data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Step 2: Train the model
    losses = train_model(model, X_train, y_train)

    # Step 3: Plot the loss
    plot_loss(losses)

    # Step 4: Evaluate
    evaluate_test_data(model, X_test, y_test)

    # Evaluation could be added here (optional)
    print("Training complete!")
