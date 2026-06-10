import numpy as np
from globals import *
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_m_output(fp = ""):

    if fp == "": # temp for testing
        rng = np.random.default_rng()

        return rng.random(size = (10000, 1000))
    
    return np.load(fp)


m1_outputs = get_m_output("data/saved_runs/m1_output.npy").squeeze()
m2_outputs = get_m_output("data/saved_runs/m2_output.npy").squeeze()
m3_outputs = get_m_output("data/saved_runs/m3_output.npy").squeeze()

def output_to_softmax(x, axis=1):
    # Subtract max for numerical stability along the specified axis
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    
    # Divide by the sum of exponentials along the same axis
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)    

# highest of prob dist
def output_to_confidence(x):
    prob = output_to_softmax(x)
    return np.max(prob, axis = 1)

# Shannon entropy
def output_to_entropy(x):
    prob = output_to_softmax(x)

    eps = 1e-12
    n = -np.sum(prob * np.log2(prob + eps), axis=1)

    return n

# difference between top 2 probabilities
def output_to_margin(x):
    prob = output_to_softmax(x)

    sorted_idx = np.argsort(prob, axis = 1)

    first_highest_key = sorted_idx[:, -1]
    second_highest_key = sorted_idx[:, -2]

    rows = np.arange(prob.shape[0])

    p1 = prob[rows, first_highest_key]
    p2 = prob[rows, second_highest_key]

    margin = p1 - p2

    return margin

def output_to_pred(x):
    return np.argmax(x, axis = 1)

def get_true_pred(pred, conf, conf_threshold):
    idk_key = conf < conf_threshold

    new_arr = pred.copy()

    new_arr[idk_key] = -1

    return new_arr

# DATA REFINEMENT

m1_conf = output_to_confidence(m1_outputs)
m2_conf = output_to_confidence(m2_outputs)

m1_entropy = output_to_entropy(m1_outputs)
m1_margin = output_to_margin(m1_outputs)
m1_pred = output_to_pred(m1_outputs)

idk_key = m1_conf < M1_P95_CONF_THRESHOLD




# DATA COLLECTION

# Training Data
# [
#  [conf, entropy, margin], 
#  ..., 
#  [conf, etnropoy, margin],
# ]
data = np.concatenate((np.expand_dims(m1_conf, axis=1), np.expand_dims(m1_entropy, axis = 1), np.expand_dims(m1_margin, axis = 1)), axis=1)
labels = m2_conf < M2_P95_CONF_THRESHOLD

# selects only values where m1 returned idk
X = data[idk_key]
y = labels[idk_key]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=40,
    min_samples_leaf=40,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Train
rf.fit(X_train, y_train)

# Predict class labels
y_pred = rf.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)

print("RF Accuracy:", acc)

joblib.dump({
    "model": rf,
    "accuracy": acc,
}, "random_forest.pkl")

# testing threshold accuracy
rows = np.arange(X_test.shape[0])

y_pred = X_test[rows, 0] < 0.3

acc = accuracy_score(y_test, y_pred)
print("Threshold Accuracy:", acc)