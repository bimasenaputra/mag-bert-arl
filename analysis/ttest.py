
import pickle
from sklearn.model_selection import KFold
import numpy as np
from scipy import stats
import itertools
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def load_list_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def k_fold_cross_validation(true, preds, k):
    kf = KFold(n_splits=k, shuffle=True)
    fold_indices = kf.split(true)
    
    for fold, (true_indices, _) in enumerate(fold_indices):
        true_fold = [true[i] for i in true_indices]
        preds_fold = [preds[i] for i in true_indices]
        yield true_fold, preds_fold

model_preds_dir = ["preds_mag-bert-base-uncased-4-t.pkl", "preds_mag-bert-base-uncased-4-ta.pkl", "preds_mag-bert-base-uncased-4-tv.pkl", "preds_mag-bert-base-uncased-6.pkl"]
k = 5

model_performance = []

for preds_dir in model_preds_dir:
    preds = load_list_from_pickle(preds_dir)
    true = load_list_from_pickle("test"+preds_dir.removeprefix("preds"))
    kfold_perf = []
    for i, (y_test, y_preds) in enumerate(k_fold_cross_validation(true, preds, k), 1):
        y_preds = np.array(y_preds) >= 5.6
        y_test = np.array(y_test) >= 5.6
        prec = precision_score(y_test, y_preds, average="weighted")
        rec = recall_score(y_test, y_preds, average="weighted")
        f_score = (2*prec*rec)/(prec+rec)
        acc = accuracy_score(y_test, y_preds)
        #print(acc)
        kfold_perf.append(acc)
    model_performance.append(kfold_perf)

models = [performance for performance in model_performance]
# Perform pairwise t-tests
num_models = len(models)
p_values = []

for i, j in itertools.combinations(range(num_models), 2):
    t_statistic, p_value = stats.ttest_ind(models[i], models[j])
    p_values.append(((i, j), p_value))

# Sort p-values
p_values.sort(key=lambda x: x[1])

# Holm-Sidak correction
alpha = 0.05
adjusted_p_values = [min(alpha / (num_models - k), 1) for k in range(num_models)]

# Print the adjusted p-values
for (model1, model2), p_value in p_values:
    print(f"Adjusted p-value for comparing model {model1 + 1} and model {model2 + 1}: {p_value}")
