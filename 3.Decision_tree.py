import pandas as pd
import numpy as np
import math
def entropy(data):
    target = data['enjoysport']
    values, counts = np.unique(target, return_counts=True)
    return sum((-count / len(target)) * math.log2(count / len(target)) for count in counts)
def information_gain(data, attribute):
    total_entropy = entropy(data)
    values, counts = np.unique(data[attribute], return_counts=True)
    weighted_entropy = sum((counts[i] / sum(counts)) * entropy(data[data[attribute] == values[i]]) 
                          for i in range(len(values)))
    return total_entropy - weighted_entropy
class DecisionTreeNode:
    def __init__(self, attribute=None, label=None, branches=None):
        self.attribute = attribute
        self.label = label
        self.branches = branches if branches is not None else {}
def id3(data, attributes):
    target = data['enjoysport']
    if len(target.unique()) == 1:
        return DecisionTreeNode(label=target.iloc[0])
    if not attributes:
        return DecisionTreeNode(label=target.value_counts().idxmax())
    gains = {attr: information_gain(data, attr) for attr in attributes}
    best_attribute = max(gains, key=gains.get)
    node = DecisionTreeNode(attribute=best_attribute) 
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value].drop(columns=[best_attribute])
        if subset.empty:
            node.branches[value] = DecisionTreeNode(label=target.value_counts().idxmax())
        else:
            new_attributes = [attr for attr in attributes if attr != best_attribute]
            node.branches[value] = id3(subset, new_attributes) 
    return node
def classify(sample, tree):
    if tree.label is not None:
        return tree.label
    attribute_value = sample.get(tree.attribute, None)
    if attribute_value not in tree.branches:
        return None  
    return classify(sample, tree.branches[attribute_value])
data = pd.read_csv("enjoysport.csv")
attributes = list(data.columns[:-1])
root = id3(data, attributes)
new_sample = {'sky': 'sunny', 'airtemp': 'warm', 'humidity': 'high', 'wind': 'strong', 'water': 'cool', 'forcast': 'change'}
prediction = classify(new_sample, root)
print(f'Prediction for the new sample: {prediction}')