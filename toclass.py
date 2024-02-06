import pickle as pkl
import numpy as np

with open("datasets/ets.pkl", "rb") as f:
    data = pkl.load(f)

data_clf = {"train": [], "dev": [], "test": []}

median = [[],[],[],[],[],[]]

for tup in data["train"]:
    for i in range(6):
        median[i].append(tup[-2][i])

for tup in data["dev"]:
    for i in range(6):
        median[i].append(tup[-2][i])

for tup in data["test"]:
    for i in range(6):
        median[i].append(tup[-2][i])

for i in range(6):
    median[i] = np.median(median[i])

print(median)

for tup in data["train"]:
    lst = list(tup)
    lst[-2] = list(lst[-2])
    for i in range(6):
        lst[-2][i] = lst[-2][i] >= median[i]
    lst[-2] = tuple(lst[-2])
    data_clf["train"].append(tuple(lst))

for tup in data["dev"]:
    lst = list(tup)
    lst[-2] = list(lst[-2])
    for i in range(6):
        lst[-2][i] = lst[-2][i] >= median[i]
    lst[-2] = tuple(lst[-2])
    data_clf["dev"].append(tuple(lst))

for tup in data["test"]:
    lst = list(tup)
    lst[-2] = list(lst[-2])
    for i in range(6):
        lst[-2][i] = lst[-2][i] >= median[i]
    lst[-2] = tuple(lst[-2])
    data_clf["test"].append(tuple(lst))

print(len(data_clf["train"]))
print(len(data_clf["test"]))
print(len(data_clf["dev"]))

#with open("datasets/ets_clf.pkl", "wb") as f:
    #pkl.dump(data_clf, f)
