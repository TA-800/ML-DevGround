import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from scipy import sparse
import matplotlib.pyplot as plt

# PREDICT WHICH HEADLINES ARE REAL AND FAKE.

# Read files
# real.txt -> real news headlines
real_dataset = pd.read_csv("real.txt", header=None, names=["Headline"])
real_dataset["Label"] = 1

# fake.txt -> fake news headlines
fake_dataset = pd.read_csv("fake.txt", header=None, names=["Headline"])
fake_dataset["Label"] = 0

# Concatenate
dataset = pd.concat([real_dataset, fake_dataset])

# CountVectorizer
vectorizer = CountVectorizer()
# Sparse matrix
X = vectorizer.fit_transform(dataset["Headline"])

# Split
train_features, test_features, train_labels, test_labels = train_test_split(
    X, dataset["Label"], test_size=0.3, shuffle=True
)

# Get validation from test
validation_features, test_features, validation_labels, test_labels = train_test_split(
    test_features, test_labels, test_size=0.5, shuffle=True
)

# Decision Tree
scores = {}

for i in range(5, 15):
    treeModel = DecisionTreeClassifier(max_depth=i, criterion="entropy")
    treeModel.fit(train_features, train_labels)
    scores[i] = treeModel.score(validation_features, validation_labels)
    print(f"Depth: {i}, Score: {scores[i]}")

min_score = min(scores.values())
best_depth = max(scores.items(), key=lambda x: x[1])[0]
best_score = scores[best_depth]

# Plot
plt.bar(scores.keys(), scores.values())
plt.ylim(min_score - 0.025, best_score + 0.025)
plt.title("Validation scores vs depths")
plt.show()

print(f"Best depth: {best_depth}")

# Train (on train + validation) with best depth
treeModel = DecisionTreeClassifier(max_depth=best_depth, criterion="entropy")
treeModel.fit(
    # Vertical stack on train and validation
    sparse.vstack([train_features, validation_features]),
    # Concatenate labels for train and validation
    pd.concat([train_labels, validation_labels]),
)
test_score = treeModel.score(test_features, test_labels)
print(test_score)

# Plot
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html
tree.plot_tree(
    treeModel,
    filled=True,
    feature_names=vectorizer.get_feature_names_out(),
    max_depth=1,
)
plt.show()
