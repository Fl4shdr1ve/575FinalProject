import pandas as pd
from collections import defaultdict, Counter

class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = defaultdict(lambda: defaultdict(lambda: 1))
        self.class_counts = Counter()
        self.total_count = 0

    def train(self, X, y):
        # X: prior actions by player and opponent
        # y: labels (C or D)
            
        self.total_count = len(y)
        self.class_counts.update(y)

        # get priors
        for cls in self.class_counts:
            self.priors[cls] = self.class_counts[cls] / self.total_count

        # calculate likelihoods
        for features, label in zip(X, y):
            for i, char in enumerate(features):
                self.likelihoods[(i, char)][label] += 1

    def predict(self, features):
        # features: string with 4 prior actions (player and opponent)
        # returns prediction (C or D)

        probs = {}

        for cls in self.priors:
            prob = self.priors[cls]

            for i, char in enumerate(features):
                prob *= self.likelihoods[(i, char)][cls]

            probs[cls] = prob

        return max(probs, key=probs.get)

# get training data
df = pd.read_csv("random_opponents.csv")
df = df[df.index % 100 != 0] # remove first game for each player b/c there are no priors

# remove last 29 games for test data
train = df[df.index % 100 <= 70]

X_train = train['prev'] + train['prev'].shift(1) # priors from previous 2 rounds
y_train = train['action_player']

X_train = X_train[X_train.index != 1]

# train
classifier = NaiveBayes()
classifier.train(X_train, y_train)

# test data - last 29 games
test = df[df.index % 100 > 70]

# append 
X_test = test['prev'] + test['prev'].shift(1)
y_test = test['action_player']

# drop first game because there are no priors
X_test = X_test[X_test.index != 71]

# test
correct = 0

for i in range(1, len(X_test)):
    if classifier.predict(X_test.iloc[i]) == y_test.iloc[i]:
        correct = correct + 1

accuracy = correct / len(X_test)

print("Accuracy: ", "{0:.3f}".format(accuracy))
