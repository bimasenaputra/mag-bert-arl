import pickle
import argparse
from argparse_utils import str2bool, seed, set_random_seed

import torch
from torch.utils.data import DataLoader

from load_dataset import get_appropriate_dataset
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE, BERT_PRETRAINED_MODEL_ARCHIVE_LIST

from model import Seq2SeqModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--num_labels", type=int, default=1)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument(
    "--model",
    type=str,
    default="bert-base-uncased",
    help="Model to train"
)
parser.add_argument(
    "--model_type",
    type=str,
    choices=["mag-bert-arl","bert-arl", "mag-bert", "bert"],
    default="mag-bert-arl",
    help="BERT model type: mag-bert-arl/bert-arl/mag-bert/bert"
)
parser.add_argument(
    "--tokenizer",
    type=str,
    choices=BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    default="bert-base-uncased",
    help="Bert tokenizer to use, see at https://huggingface.co/models?filter=bert"
)
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the learner")
parser.add_argument("--gradient_accumulation_step", type=int, default=1, help="")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="")
parser.add_argument("--seed", type=seed, default="random", help="Seed for reproducibility")
parser.add_argument("--no_save", type=str2bool, default="false", help="Save model every epoch")
#parser.add_argument("--with_question", type=str2bool, default="true", help="Question context")
parser.add_argument("--lr_adversary", type=float, default=1e-5, help="Learning rate for the adversary")
parser.add_argument("--pretrain_steps", type=int, default=250, help="Number of steps to pretrain the learner")
args = parser.parse_args()
args.device = DEVICE

def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    train_dataset = get_appropriate_dataset(train_data, args.max_seq_length, args.tokenizer, VISUAL_DIM, ACOUSTIC_DIM)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_data = data["dev"]
    if dev_data is not None:
        dev_dataset = get_appropriate_dataset(dev_data, args.max_seq_length, args.tokenizer, VISUAL_DIM, ACOUSTIC_DIM)
        dev_dataloader = DataLoader(
            dev_dataset, batch_size=args.dev_batch_size, shuffle=True
        )
    else:
        dev_dataloader = None

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )

    return (
        train_dataloader,
        dev_dataloader,
        num_train_optimization_steps,
    )

def main():
    (
        train_data_loader,
        dev_data_loader,
        args.num_train_optimization_steps,
    ) = set_up_data_loader()
    model = Seq2SeqModel(args.model_type, args.model, args)
    model.train(train_data_loader, dev_data_loader)


if __name__ == "__main__":
    main()