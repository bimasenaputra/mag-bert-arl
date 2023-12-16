import numpy as np
from typing import *
from argparse_utils import set_random_seed

import torch
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, DataParallel

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from bert import MAG_BertForSequenceClassification, MAG_BertWithARL

from tqdm import tqdm, trange

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob

class Seq2SeqModel:
    def __init__(
        self,
        model_type=None,
        model_name=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        **kwargs
    ):

        """
        Initializes a Seq2SeqModel.

        Args:
            model_type (optional): The type of model to use as the encoder.
            model_name (optional): The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        self.args = args

        if model_type is None and args.model_type is not None:
            model_type = args.model_type

        if model_name is None and args.model is not None:
            model_name = args.model

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            self.device = "cpu"

        set_random_seed(args.seed)

        if not use_cuda:
            self.args.fp16 = False

        multimodal_config = MultimodalConfig(
            beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
        )

        # TODO: bert-arl
        if model_type is "bert":
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=self.args.num_labels)
        elif model_type is "mag-bert":
            self.model = MAG_BertForSequenceClassification.from_pretrained(model_name, multimodal_config=multimodal_config, num_labels=self.args.num_labels)
        elif model_type is "mag-bert-arl":
            self.model = MAG_BertWithARL.from_pretrained(model_name, multimodal_config=multimodal_config, num_labels=self.args.num_labels)
        else:
            pass

        self.model.to(self.device)

        if args.n_gpu > 1:
            self.model = DataParallel(self.model)

    def prep_for_training():
        # TODO: adversary optimizer
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if ARL:
            assert args.train_batch_size == args.dev_batch_size

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_proportion * self.args.num_train_optimization_steps,
            num_training_steps=self.args.num_train_optimization_steps,
        )
        return optimizer, scheduler


    def train_epoch(train_dataloader: DataLoader, optimizer, scheduler):
        self.model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = self.model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            logits = outputs[0]
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if self.args.gradient_accumulation_step > 1:
                loss = loss / self.args.gradient_accumulation_step

            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()

            tr_loss += loss.item()
            nb_tr_steps += 1

            if (step + 1) % self.args.gradient_accumulation_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        return tr_loss / nb_tr_steps


    def eval_epoch(dev_dataloader: DataLoader, optimizer):
        self.model.eval()
        dev_loss = 0
        nb_dev_examples, nb_dev_steps = 0, 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)

                input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
                visual = torch.squeeze(visual, 1)
                acoustic = torch.squeeze(acoustic, 1)
                outputs = self.model(
                    input_ids,
                    visual,
                    acoustic,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )
                logits = outputs[0]

                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if self.args.gradient_accumulation_step > 1:
                    loss = loss / self.args.gradient_accumulation_step

                if args.n_gpu > 1:
                    loss = loss.mean()

                dev_loss += loss.item()
                nb_dev_steps += 1

        return dev_loss / nb_dev_steps

    def test_epoch(test_dataloader: DataLoader):
        self.model.eval()
        preds = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                batch = tuple(t.to(self.device) for t in batch)

                input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
                visual = torch.squeeze(visual, 1)
                acoustic = torch.squeeze(acoustic, 1)
                outputs = self.model(
                    input_ids,
                    visual,
                    acoustic,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )

                logits = outputs[0]

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.detach().cpu().numpy()

                logits = np.squeeze(logits).tolist()
                label_ids = np.squeeze(label_ids).tolist()

                preds.extend(logits)
                labels.extend(label_ids)

            preds = np.array(preds)
            labels = np.array(labels)

        return preds, labels


    def test_score_model(test_dataloader: DataLoader, use_zero=False):

        preds, y_test = test_epoch(test_dataloader)
        non_zeros = np.array(
            [i for i, e in enumerate(y_test) if e != 0 or use_zero])

        preds = preds[non_zeros]
        y_test = y_test[non_zeros]

        mae = np.mean(np.absolute(preds - y_test))
        corr = np.corrcoef(preds, y_test)[0][1]

        preds = preds >= 0
        y_test = y_test >= 0

        f_score = f1_score(y_test, preds, average="weighted")
        acc = accuracy_score(y_test, preds)

        return acc, mae, corr, f_score

    def test(test_data_loader):
        test_accuracies = []

        for epoch_i in range(int(self.args.n_epochs)):
            test_acc, test_mae, test_corr, test_f_score = test_score_model(test_data_loader)

            print(
                "epoch:{}, test_acc:{}".format(
                    epoch_i, test_acc
                )
            )

            test_accuracies.append(test_acc)

    def train(train_dataloader, validation_dataloader=None):
        optimizer, scheduler = prep_for_training()
        valid_losses = []

        for epoch_i in range(int(self.args.n_epochs)):
            train_loss = train_epoch(train_dataloader, optimizer, scheduler)
            valid_loss = eval_epoch(validation_dataloader, optimizer) if validation_dataloader is not None else 0.0

            print(
                "epoch:{}, train_loss:{}, valid_loss:{}".format(
                    epoch_i, train_loss, valid_loss
                )
            )

            valid_losses.append(valid_loss)