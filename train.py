import pickle
import argparse
from argparse_utils import str2bool, seed, set_random_seed

import torch
from torch.utils.data import DataLoader

from load_dataset import get_appropriate_dataset
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE, BERT_PRETRAINED_MODEL_ARCHIVE_LIST

from model import Seq2SeqModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei", "ets", "fiv2"], default="fiv2", help="Pickle (.pkl) file saved in ./dataset folder in { train: {(), }, test: {...}, dev (optional): {...}} format")
parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum number of tokens the model can take in a single input")
parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--num_labels", type=int, default=1, help="Number of classes/labels to predict")
parser.add_argument("--dev_batch_size", type=int, default=64, help="Batch size for dev/eval")
parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--beta_shift", type=float, default=1.0, help="The constant 'beta' to be used in the adaption gate during feature fusion with other features")
parser.add_argument("--dropout_prob", type=float, default=0.5, help="Probability of a neuron being dropped out during each training session")
parser.add_argument("--model", type=str, default="bert-base-uncased", help="Name of model to train")
parser.add_argument("--model_type", type=str, choices=["mag-bert-arl", "bert-arl", "mag-bert"], default="mag-bert-arl", help="MAG-BERT model type")
parser.add_argument("--tokenizer", type=str, choices=BERT_PRETRAINED_MODEL_ARCHIVE_LIST, default="bert-base-uncased", help="Bert tokenizer to use")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the learner")
parser.add_argument("--gradient_accumulation_step", type=int, default=1, help="Number of steps before gradients are summed to update weights")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Proportion of training used for warmup")
parser.add_argument("--seed", type=seed, default=8, help="Seed for reproducibility")
parser.add_argument("--no_save", type=str2bool, default="false", help="Save model at the end of training")
parser.add_argument("--save_model_every_epoch", type=str2bool, default="false", help="Save model every epoch")
parser.add_argument("--output_dir", type=str, default="mag-bert-base-uncased-3", help="Output directory where the model will be saved")
parser.add_argument("--learning_rate_adversary", type=float, default=1e-5, help="Learning rate for the adversary")
parser.add_argument("--pretrain_steps", type=int, default=6, help="Number of steps to pretrain the learner, -1 means no pretraining")
parser.add_argument("--use_early_stopping", type=str2bool, default="false", help="Whether or not to use early stopping to mitigate too many epochs")
parser.add_argument("--early_stopping_patience", type=int, default=2, help="Number of steps until there is no improvement in validation loss")
parser.add_argument("--use_cuda", type=str2bool, default="true", help="Use cuda for training")
parser.add_argument("--cuda_device", type=int, default=0, help="Cuda device to use for training")
parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPU used for training")
parser.add_argument("--wandb", type=str, default="", help="Name of WANDB project")
args = parser.parse_args()
args.device = DEVICE

def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    train_dataset = get_appropriate_dataset(train_data, args.max_seq_length, args.tokenizer, VISUAL_DIM, ACOUSTIC_DIM)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
    )

    dev_data = data["dev"]
    if dev_data is not None:
        dev_dataset = get_appropriate_dataset(dev_data, args.max_seq_length, args.tokenizer, VISUAL_DIM, ACOUSTIC_DIM)
        dev_dataloader = DataLoader(
            dev_dataset, batch_size=args.dev_batch_size, shuffle=True, drop_last=True,
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
        train_dataloader,
        dev_dataloader,
        args.num_train_optimization_steps,
    ) = set_up_data_loader()
    model = Seq2SeqModel(args.model_type, args.model, args, args.use_cuda, args.cuda_device)
    model.train(train_dataloader, dev_dataloader)


if __name__ == "__main__":
    main()
