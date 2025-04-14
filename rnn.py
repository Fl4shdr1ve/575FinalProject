import torch
import torch.nn as nn
import pandas as pd

df = pd.read_csv('fix.csv')
df = df.drop(columns=['player', 'opponent', 'payoff', 'time_php', 'time_js', 'treatment', 'session', 'context', 'prev', 'prev_player', 'prev_opp'])

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


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.linear(out[:, -1, :]) 

        return out

model = RNN(input_size=2, hidden_size=128, output_size=2)


line = df.head(99)['action_player'].to_string(index=False).replace('\n', '')

input = lineToTensor(line)

output = model(input)
print(output)