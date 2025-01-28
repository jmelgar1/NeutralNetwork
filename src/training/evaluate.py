import torch
from matplotlib import pyplot as plt


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