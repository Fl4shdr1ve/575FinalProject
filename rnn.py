import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc

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
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        out, _ = self.rnn(x, h)
        out = out[:, -1, :]
        out = self.linear(out)
        
        return self.sigmoid(out)


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

# hyperparameters
input_size = 2
hidden_size = 8
output_size = 1
seq_len = 4
num_layers = 2
batch_size = len(X_train)
num_epochs = 100
learning_rate = 0.01

# turn into tensors
X_train_tensor = torch.zeros(batch_size, seq_len, input_size)

for i, line in enumerate(X_train):
    X_train_tensor[i] = lineToTensor(line)[0]
    
y_train_tensor = torch.zeros(batch_size, 1)

for i, line in enumerate(y_train):
    y_train_tensor[i] = lineToTensor(line)[0][0][0]
    

X_train = X_train_tensor
y_train = y_train_tensor

model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
for epoch in range(num_epochs):
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
       
print(outputs)
      
# test set
X_test = X[X.index % 100 > 70]
y_test = y[y.index % 100 > 70]

# turn into tensors
X_test_tensor = torch.zeros(len(X_test), seq_len, input_size)

for i, line in enumerate(X_test):
    X_test_tensor[i] = lineToTensor(line)[0]
    
y_test_tensor = torch.zeros(len(X_test), 1)

for i, line in enumerate(y_test):
    y_test_tensor[i] = lineToTensor(line)[0][0][0]
    

X_test = X_test_tensor
y_test = y_test_tensor
        
        
# test
model.eval()

with torch.no_grad():
    outputs = torch.round(model(X_test))
    correct = (outputs == y_test).sum().item()
    accuracy = correct / len(X_test)
    
    print("Accuracy: ", "{0:.3f}".format(accuracy))


# ROC Curve
"""
with torch.no_grad():
    y_probs = model(X_train).numpy().flatten()
    y_true = y_train.numpy().flatten()

fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Recurrent Neural Network')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("RNN_ROC")
plt.show()
"""