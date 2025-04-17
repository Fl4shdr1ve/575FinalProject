import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class NaiveBayes:
    def __init__(self):
        self.class_counts = Counter()
        self.priors = {}
        self.likelihoods = defaultdict(lambda: defaultdict(lambda: 1))
        self.total_count = 0

    def train(self, X, y):
        self.total_count = len(y)
        self.class_counts.update(y)

        # get priors
        for cls in self.class_counts:
            self.priors[cls] = self.class_counts[cls] / self.total_count

        # calculate likelihoods
        for features, label in zip(X, y):
            for i, char in enumerate(features):
                self.likelihoods[(i, char)][label] += 1

    def predict(self, features): # returns C or D
        probs = {}

        for cls in self.priors:
            prob = self.priors[cls]

            for i, char in enumerate(features):
                prob *= self.likelihoods[(i, char)][cls]

            probs[cls] = prob

        return max(probs, key=probs.get)
    
    def predict_prob(self, features): # gets probability of a defect (D)
        probs = {}
        
        for cls in self.priors:
            prob = self.priors[cls]
            
            for i, char in enumerate(features):
                prob *= self.likelihoods[(i, char)][cls]
                
            probs[cls] = prob

        total = sum(probs.values())
        
        for cls in probs:
            probs[cls] /= total  # divide by total to make probability
            
        return probs

# get training data
df = pd.read_csv("random_opponents.csv")
df = df[df.index % 100 != 0] # remove first game for each player b/c there are no priors

X = df['prev'] + df['prev'].shift(1)
y = df['action_player']

# remove last 29 games in each set from train for test
# cannot use train_test_split because the games are sequential and there are multiple different players
"""
train = df[df.index % 100 <= 70]

X_train = train['prev'] + train['prev'].shift(1) # priors from previous 2 rounds
y_train = train['action_player']
"""

X_train = X[X.index % 100 <= 70]
y_train = y[y.index % 100 <= 70]

# remove first game for each player because there are no priors
X_train = X_train[X_train.index % 100 != 1]
y_train = y_train[y_train.index % 100 != 1]

# train
classifier = NaiveBayes()
classifier.train(X_train, y_train)

# test data - last 29 games of each 100
X_test = X[X.index % 100 > 70]
y_test = y[y.index % 100 > 70]

# test
correct = 0

for i in range(1, len(X_test)):
    if classifier.predict(X_test.iloc[i]) == y_test.iloc[i]:
        correct = correct + 1

accuracy = correct / len(X_test)

print("Accuracy: ", "{0:.3f}".format(accuracy))


# ROC Curve 
"""
y_true = [1 if y == 'D' else 0 for y in y_test]
y_scores = [classifier.predict_prob(x)['D'] for x in X_test]

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naive Bayes Classifier')
plt.legend(loc="lower right")
plt.grid()
plt.savefig("Naive_Bayes_ROC")
plt.show()
"""