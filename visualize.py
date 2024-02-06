import argparse
import pickle
import torch

from torch.utils.data import DataLoader
from typing import Any
from transformers import BertTokenizer
from bert import MAG_BertForSequenceClassification, MAG_BertWithARL
from load_dataset import get_appropriate_dataset
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE, BERT_PRETRAINED_MODEL_ARCHIVE_LIST
from captum.attr import (GradientShap, configure_interpretable_embedding_layer)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei", "ets"], default="ets", help="Pickle (.pkl) file saved in ./dataset folder in { train: {(), }, test: {...}, dev (optional): {...}} format")
parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum number of tokens the model can take in a single input")
parser.add_argument("--beta_shift", type=float, default=1.0, help="The constant 'beta' to be used in the adaption gate during feature fusion with other features")
parser.add_argument("--dropout_prob", type=float, default=0.5, help="Probability of a neuron being dropped out during each training session")
parser.add_argument("--model", type=str, default="bert-base-uncased", help="Name of model to train")
parser.add_argument("--model_type", type=str, choices=["mag-bert-arl", "mag-bert"], default="mag-bert-arl", help="MAG-BERT model type")
parser.add_argument("--num_labels", type=int, default=1, help="Number of classes/labels to predict")
parser.add_argument("--tokenizer", type=str, choices=BERT_PRETRAINED_MODEL_ARCHIVE_LIST, default="bert-base-uncased", help="Bert tokenizer to use")
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

model = model.eval()
model.to(DEVICE)

def visualize_children(
    object : Any,
    level : int = 0,
) -> None:
    """
    Prints the children of (object) and their children too, if there are any.
    Uses the current depth (level) to print things in a ordonnate manner.
    """
    print(f"{'   ' * level}{level}- {type(object).__name__}")
    try:
        for child in object.children():
            visualize_children(child, level + 1)
    except:
        pass

shap_log = {"t": [], "v":[], "a":[]}
with open("shap.pkl", "wb") as f:
    pickle.dump(shap_log, f)

tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
    data = pickle.load(handle)

train_data = data["train"]
train_dataset = get_appropriate_dataset(train_data, args.max_seq_length, args.tokenizer, VISUAL_DIM, ACOUSTIC_DIM)
train_dataloader = DataLoader(
        train_dataset, shuffle=True, drop_last=True,
    )
gradient_shap = GradientShap(model)
intr = configure_interpretable_embedding_layer(model, "bert.embeddings")

for tup in train_dataloader:
    print(len(tup))
    tup = tuple(t.to(DEVICE) for t in tup)
    input_ids, visual, acoustic, input_mask, segment_ids, label_ids = tup
    #print(type(input_ids))
    visual = torch.squeeze(visual, 1)
    acoustic = torch.squeeze(acoustic, 1)
    outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=label_ids,
            )
    target = outputs[2]
    #print(target)
    rndm = torch.rand(input_ids.size()).to(DEVICE)
    alpha = torch.rand(input_ids.size()).to(DEVICE)
    random_point = rndm + alpha * (input_ids - rndm)
    input_ids = intr.indices_to_embeddings(input_ids).unsqueeze(0)
    visual = intr.indices_to_embeddings(visual)
    acoustic = intr.indices_to_embeddings(acoustic)
    random_point = intr.indices_to_embeddings(random_point).unsqueeze(0)
    inpt = (input_ids, visual, acoustic)
    baseline = (random_point, visual * 0, acoustic * 0)
    attribution = gradient_shap.attribute(inpt, baseline, target=target)
    shap_log["t"].append(attributions[0].sum().item())
    shap_log["v"].append(attributions[1].sum().item())
    shap_log["a"].append(attributions[2].sum().item())
    print('Text Contributions: ', attributions[0].sum().item(), 'Visual Contributions: ', attributions[1].sum().item(), 'Acoustic Contributions: ', attributions[2].sum().item())

with open("shap.pkl", "wb") as f:
    pickle.dump(shap_log, f)
#visualize_children(model)
