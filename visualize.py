import argparse
import pickle
import torch

from torch.utils.data import DataLoader
from typing import Any
from transformers import BertTokenizer
from bert import MAG_BertForSequenceClassification, MAG_BertWithARL
from load_dataset import get_appropriate_dataset
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE, BERT_PRETRAINED_MODEL_ARCHIVE_LIST
from captum.attr import GradientShap, configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper

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


torch.backends.cudnn.enabled=False
model = ModelInputWrapper(model)
model = torch.nn.DataParallel(model)
model.eval()
model.to(DEVICE)

def wrapped_model(input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None
    ):
    return model(input_ids,
        visual,
        acoustic,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )[0]


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

intr_embed = configure_interpretable_embedding_layer(model, "module.module.bert.embeddings")
gradient_shap = GradientShap(wrapped_model)

for tup in train_dataloader:
    tup = tuple(t.to(DEVICE) for t in tup)
    input_ids, visual, acoustic, input_mask, segment_ids, label_ids = tup
    visual = torch.squeeze(visual, 1)
    acoustic = torch.squeeze(acoustic, 1)
    rndm = torch.rand(input_ids.size()).to(DEVICE).long()
    alpha = torch.rand(input_ids.size()).to(DEVICE).long()
    random_point = (rndm + alpha * (input_ids - rndm)).to(DEVICE).long()
    input_ids = intr_embed.indices_to_embeddings(input_ids)
    random_point = intr_embed.indices_to_embeddings(random_point)
    inpt = (input_ids, visual, acoustic)
    baseline = (random_point, visual * 0, acoustic * 0)
    attributions = gradient_shap.attribute(inpt, baseline, target=0, additional_forward_args=(input_mask, segment_ids))
    shap_log["t"].append(attributions[0].sum().item())
    shap_log["v"].append(attributions[1].sum().item())
    shap_log["a"].append(attributions[2].sum().item())
    print('Text Contributions: ', attributions[0].sum().item(), 'Visual Contributions: ', attributions[1].sum().item(), 'Acoustic Contributions: ', attributions[2].sum().item())

with open("shap.pkl", "wb") as f:
    pickle.dump(shap_log, f)
