import pickle as pkl

with open("datasets/ets.pkl", "rb") as f:
    data = pkl.load(f)

print(data["train"][0][-1])
print(type(data["train"]))
