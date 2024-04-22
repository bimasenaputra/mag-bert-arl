import pickle as pkl
import pandas as pd

csv = pd.read_csv('analysis/fiv2_gender_pred.csv', delimiter=';', skipinitialspace=True)
print(csv.head())

def getGender(filename):
    filename = filename + ".mp4"
    print(filename)
    row = csv[csv['VideoName'] == filename]
    try:
        gender = row.values.tolist()[0][-1]
    except IndexError:
        gender = 1
    return gender

with open("datasets/fi.pkl", "rb") as f:
    data = pkl.load(f)

data_male = {"train": [], "dev": [], "test": []}
data_female = {"train": [], "dev": [], "test": []}

for tup in data["train"]:
    gender = getGender(tup[-1])
    if gender == 1:
        data_male["train"].append(tup)
    elif gender == 2:
        data_female["train"].append(tup)

for tup in data["dev"]:
    gender = getGender(tup[-1])
    if gender == 1:
        data_male["dev"].append(tup)
    elif gender == 2:
        data_female["dev"].append(tup)

for tup in data["test"]:
    gender = getGender(tup[-1])
    if gender == 1:
        data_male["test"].append(tup)
    elif gender == 2:
        data_female["test"].append(tup)

with open("datasets/fi_male.pkl", "wb") as f:
    pkl.dump(data_male, f)

with open("datasets/fi_female.pkl", "wb") as f:
    pkl.dump(data_female, f)


