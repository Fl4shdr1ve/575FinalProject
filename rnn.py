import torch
import torch.nn as nn
import pandas as pd

# encode df actions (action_player, and action_opponent)
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, 2)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def letterToIndex(letter):
    if letter == 'C':
        return 0
    elif letter =='D':
        return 1

"""
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.linear(out[:, -1, :]) 

        return out

model = RNN(input_size=2, hidden_size=128, output_size=1)
"""

# get data
df = pd.read_csv("random_opponents.csv")
df = df[df.index % 100 != 0] # remove first game for each player b/c there are no priors

X = df['prev'] + df['prev'].shift(1)
y = df['action_player']

X_train = X[X.index % 100 <= 70]
y_train = y[y.index % 100 <= 70]

# remove first game for each player because there are no priors
X_train = X_train[X_train.index % 100 != 1]
y_train = y_train[y_train.index % 100 != 1]

# turn into tensors
X_train = X_train.apply(lineToTensor)
y_train = y_train.apply(lineToTensor)

# Define hyperparameters
input_size = 2  # Number of features in the input
hidden_size = 20 # Number of features in the hidden state
num_layers = 1   # Number of RNN layers
seq_len = 4      # Length of the input sequence
batch_size = len(X_train)   # Number of sequences in a batch

# Create an RNN instance
rnn = nn.RNN(input_size, hidden_size, num_layers)

# Create a random input tensor
print(X_train)
input_tensor = X_train
rand_tensor = torch.randn(seq_len, batch_size, input_size)
print(rand_tensor)

# Initialize the hidden state (optional, defaults to zero if not provided)
hidden_state = torch.zeros(num_layers, batch_size, hidden_size)

# Pass the input tensor and hidden state through the RNN
output, hidden_state = rnn(input_tensor, hidden_state)

print("Output shape:", output.shape)
print("Hidden state shape:", hidden_state.shape)