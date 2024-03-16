import pickle as pkl
import pandas as pd

csv = pd.read_csv('GENDER_pred.csv', delimiter=',', skipinitialspace=True)

def getGender(filename):
    filename = filename + ".mp4"
    row = csv[csv['Filename'] == filename]
    gender = row.values.tolist()[0][1]
    
    return gender

with open("datasets/ets.pkl", "rb") as f:
    data = pkl.load(f)

data_male = {"train": [], "dev": [], "test": []}
data_female = {"train": [], "dev": [], "test": []}

for tup in data["train"]:
    gender = getGender(tup[-1])
    if gender == "male":
        data_male["train"].append(tup)
    elif gender == "female":
        data_female["train"].append(tup)

for tup in data["dev"]:
    gender = getGender(tup[-1])
    if gender == "male":
        data_male["dev"].append(tup)
    elif gender == "female":
        data_female["dev"].append(tup)

for tup in data["test"]:
    gender = getGender(tup[-1])
    if gender == "male":
        data_male["test"].append(tup)
    elif gender == "female":
        data_female["test"].append(tup)

with open("datasets/ets_male.pkl", "wb") as f:
    pkl.dump(data_male, f)

with open("datasets/ets_female.pkl", "wb") as f:
    pkl.dump(data_female, f)


