import argparse
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from bert import MAG_BertForSequenceClassification, MAG_BertWithARL
from load_dataset import get_appropriate_dataset
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE, BERT_PRETRAINED_MODEL_ARCHIVE_LIST
from argparse_utils import str2bool

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei", "ets", "fi"], default="ets", help="Pickle (.pkl) file saved in ./dataset folder in { train: {(), }, test: {...}, dev (optional): {...}} format")
parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum number of tokens the model can take in a single input")
parser.add_argument("--beta_shift", type=float, default=1.0, help="The constant 'beta' to be used in the adaption gate during feature fusion with other features")
parser.add_argument("--dropout_prob", type=float, default=0.5, help="Probability of a neuron being dropped out during each training session")
parser.add_argument("--model", type=str, default="mag-bert-base-uncased-3", help="Name of model to train")
parser.add_argument("--model_type", type=str, choices=["mag-bert-arl", "mag-bert"], default="mag-bert-arl", help="MAG-BERT model type")
parser.add_argument("--num_labels", type=int, default=1, help="Number of classes/labels to predict")
parser.add_argument("--tokenizer", type=str, choices=BERT_PRETRAINED_MODEL_ARCHIVE_LIST, default="bert-base-uncased", help="Bert tokenizer to use")
parser.add_argument("--use_cuda", type=str2bool, default="true", help="Use cuda")
args = parser.parse_args()
args.device = DEVICE

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob

multimodal_config = MultimodalConfig(
    beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
)
if args.model_type == "mag-bert":
    model = MAG_BertForSequenceClassification.from_pretrained(args.model, multimodal_config=multimodal_config, num_labels=args.num_labels)
elif args.model_type == "mag-bert-arl":
    model = MAG_BertWithARL.from_pretrained(args.model, multimodal_config=multimodal_config, num_labels=args.num_labels)

model.eval()
model.to(args.device)

with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
    data = pickle.load(handle)

gender = pd.read_csv("analysis/ets_gender_pred.csv")
gender = dict(zip(gender['Filename'], gender['Predicted_Label']))    
gender_map = {"female": 0, "male": 1}
#gender = pd.read_csv("analysis/fiv2_gender_pred.csv", sep=";")
#gender = dict(zip(gender['VideoName'], gender['Gender']))    
#gender_map = {1: 1, 2: 0}

#race = pd.read_csv("analysis/ets_race_pred.csv")
#race = dict(zip(race['Filename'], race['Race']))
#race_map = {'white': 2, 'black': 3, 'asian': 4, 'latino hispanic': 5, 'middle eastern': 6, 'indian': 7}

test_data = data["test"]
test_dataset = get_appropriate_dataset(test_data, args.max_seq_length, args.tokenizer, VISUAL_DIM, ACOUSTIC_DIM)
test_dataloader = DataLoader(
        test_dataset, batch_size=1, drop_last=False,
    )

def fair_metrics():
    preds = []
    labels = []
    gender_sf = []
    #race_sf = []
    with torch.no_grad():
        for idx, tup in enumerate(test_dataloader,0):
            tup = tuple(t.to(args.device) for t in tup)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = tup
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            logits = outputs[0].detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            # ets 5.6, fi 0.5
            logits = np.squeeze(logits).tolist() >= 5.6
            label_ids = np.squeeze(label_ids).tolist() >= 5.6
            pred_gender = gender_map[gender[test_data[idx][-1]+".mp4"]]
            #pred_race = race_map[race.get(test_data[idx][-1]+'.jpg', 'white')]

            preds.append(logits)
            labels.append(label_ids)
            gender_sf.append(pred_gender)
            #race_sf.append(pred_race)

    preds = np.array(preds)
    labels = np.array(labels)
    gender_sf = np.array(gender_sf)
    #race_sf = np.array(race_sf)
    #s_f_frame = pd.DataFrame(np.stack([gender_sf, race_sf], axis=1), columns=['Gender', 'Race'])

    spdd = demographic_parity_difference(labels, preds, sensitive_features=gender_sf)
    eodd = equalized_odds_difference(labels, preds, sensitive_features=gender_sf)
    #ep = equal_opportunity_difference(labels, preds, sensitive_features=s_f_frame)

    return spdd, eodd

print(fair_metrics())
