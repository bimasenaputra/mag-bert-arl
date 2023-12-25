import argparse
from argparse_utils import str2bool, seed

import torch
from torch.utils.data import DataLoader

from argparse_utils import str2bool, seed, set_random_seed
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE, BERT_PRETRAINED_MODEL_ARCHIVE_LIST

from model import Seq2SeqModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--num_labels", type=int, default=1)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--model", type=str, default="bert-base-uncased", help="Name of model to train")
parser.add_argument("--model_type", type=str, choices=["mag-bert-arl","bert-arl", "mag-bert", "bert"], default="mag-bert-arl", help="BERT model type")
parser.add_argument("--tokenizer", type=str, choices=BERT_PRETRAINED_MODEL_ARCHIVE_LIST, default="bert-base-uncased", help="Bert tokenizer to use")
parser.add_argument("--seed", type=seed, default="random", help="Seed for reproducibility")
args = parser.parse_args()
args.device = DEVICE

def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    test_data = data["test"]

    test_dataset = get_appropriate_dataset(test_data, args.max_seq_length, args.tokenizer, VISUAL_DIM, ACOUSTIC_DIM)

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True
    )

    return test_dataloader

def main():
    model = Seq2SeqModel(args.model_type, args.model, args)
    test_data_loader = set_up_data_loader()
    model.test(test_data_loader)


if __name__ == "__main__":
    main()