import torch
import pandas as pd

from src.model.model import Model
from src.training.train import train

url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
df = pd.read_csv(url)

def main():
    # Set the random seed for reproducibility
    torch.manual_seed(41)

    # Train it
    train(Model(), df)

main()