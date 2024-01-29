import os
import logging
import numpy as np
from typing import *
from argparse_utils import set_random_seed

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score

from transformers import get_linear_schedule_with_warmup
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
            model_type (optional): The type of model to use.
            model_name (optional): The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            args (optional): A dict containing the args that should be changed in the default args. All the default args are available in the training and testing file as a command input. 
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        self.args = args

        if model_type is None and args.model_type is not None:
            model_type = args.model_type

        if model_name is None and args.model is not None:
            model_name = args.model

        self.model_type = model_type
        self.model_name = model_name

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

        multimodal_config = MultimodalConfig(
            beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
        )

        if model_type == "mag-bert":
            self.model = MAG_BertForSequenceClassification.from_pretrained(model_name, multimodal_config=multimodal_config, num_labels=self.args.num_labels)
        elif model_type == "mag-bert-arl":
            self.model = MAG_BertWithARL.from_pretrained(model_name, multimodal_config=multimodal_config, num_labels=self.args.num_labels)

        self.model.to(self.device)

        if args.n_gpu > 1:
            self.model = DataParallel(self.model)

    def get_learner_optimizer(self):
        # Step 1: Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        # Exclude adversary parameters if it's an ARL model
        adversary_param_optimizer = list(self.model.get_adversary_named_parameters()) if self.model_type == "mag-bert-arl" else []   
        no_grad = [n for n,p in adversary_param_optimizer]

        optimizer_grouped_parameters = [
            {
                "params": [ 
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(ng in n for ng in no_grad)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not any(ng in n for ng in no_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_proportion * self.args.num_train_optimization_steps,
            num_training_steps=self.args.num_train_optimization_steps,
        )

        # Step 2: Load model if previous training existed
        if (
            self.model_name
            and os.path.isfile(os.path.join(self.model_name, "optimizer.pt"))
            and os.path.isfile(os.path.join(self.model_name, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(self.model_name, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.model_name, "scheduler.pt")))
            
        return optimizer, scheduler

    def get_adversary_optimizer(self):
        if self.model_type != "mag-bert-arl":
            return None, None

        # Step 1: Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        # Exclude learner parameters
        learner_param_optimizer = list(self.model.get_learner_named_parameters())
        no_grad = [n for n,p in learner_param_optimizer]

        optimizer_grouped_parameters = [
            {
                "params": [ 
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(ng in n for ng in no_grad)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not any(ng in n for ng in no_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        num_train_optimization_steps_adv = self.args.num_train_optimization_steps - int(self.args.pretrain_steps/self.args.gradient_accumulation_step) * self.args.n_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate_adversary)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_proportion * num_train_optimization_steps_adv,
            num_training_steps=num_train_optimization_steps_adv,
        )

        # Step 2: Load model if previous training existed
        if (
            self.model_name
            and os.path.isfile(os.path.join(self.model_name, "optimizer-adv.pt"))
            and os.path.isfile(os.path.join(self.model_name, "scheduler-adv.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(self.model_name, "optimizer-adv.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.model_name, "scheduler-adv.pt")))
            
        return optimizer, scheduler

    def load_last_checkpoint(self, train_dataloader_len):
        global_step = 0 
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if self.model_name and os.path.exists(self.model_name):
            try:
                # set global_step to global_step of last saved checkpoint from model path
                # specify model args as model-name/checkpoint-x-epoch-y to load model from last checkpoint
                checkpoint_suffix = self.model_name.split("/")[-1].split("-")
                global_step = int(checkpoint_suffix[1])
                epochs_trained = global_step // (train_dataloader_len // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                    train_dataloader_len // self.args.gradient_accumulation_steps
                )

                logging.info("Continuing training from checkpoint, will skip to saved global_step")
                logging.info("Continuing training from epoch %d", epochs_trained)
                logging.info("Continuing training from global step %d", global_step)
                logging.info("Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
            except ValueError:
                logging.info("Starting fine-tuning.")

        return global_step, epochs_trained, steps_trained_in_current_epoch

    def train_epoch(self, train_dataloader: DataLoader, optimizer, scheduler, global_step, epoch_number, steps_trained_in_current_epoch=0, adv_optimizer=None, adv_scheduler=None):
        self.model.train()
        global_step_new = global_step
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            # adv_optimizer and adv_scheduler both shouldn't be none if and only if model has/uses ARL.
            if adv_optimizer and adv_scheduler and step > self.args.pretrain_steps:
                self.model.set_pretrain(False)
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            label_ids = label_ids.view(self.args.train_batch_size, self.args.num_labels)
            outputs = self.model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=label_ids,
            )
            loss = outputs[0]

            if self.args.gradient_accumulation_step > 1:
                loss = loss / self.args.gradient_accumulation_step

            if self.args.n_gpu > 1:
                loss = loss.mean()

            loss.backward(retain_graph=True)

            if adv_optimizer and adv_scheduler:
                adv_loss = outputs[1]

                if self.args.gradient_accumulation_step > 1:
                    adv_loss = adv_loss / self.args.gradient_accumulation_step

                if self.args.n_gpu:
                    adv_loss = adv_loss.mean()

                if step > self.args.pretrain_steps:
                    adv_loss.backward()

            tr_loss += loss.item()
            nb_tr_steps += 1

            if (step + 1) % self.args.gradient_accumulation_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if adv_optimizer and adv_scheduler and step > self.args.pretrain_steps:
                    adv_optimizer.step()
                    adv_scheduler.step()
                    adv_optimizer.zero_grad()

                global_step_new += 1

        if self.args.save_model_every_epoch:
            output_dir_current = os.path.join(self.model_name, "checkpoint-{}-epoch-{}".format(global_step_new, epoch_number))
            os.makedirs(output_dir_current, exist_ok=True)
            self.save_model(output_dir_current, optimizer, scheduler, adv_optimizer, adv_scheduler)

        if adv_optimizer and adv_scheduler:
            self.model.set_pretrain(True)

        return tr_loss / nb_tr_steps, global_step_new


    def eval_epoch(self, dev_dataloader: DataLoader):
        self.model.eval()
        dev_loss = 0
        nb_dev_examples, nb_dev_steps = 0, 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)

                input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
                visual = torch.squeeze(visual, 1)
                acoustic = torch.squeeze(acoustic, 1)
                label_ids = label_ids.view(self.args.dev_batch_size, self.args.num_labels)
                outputs = self.model(
                    input_ids,
                    visual,
                    acoustic,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=label_ids,
                )
                loss = outputs[0]

                if self.args.gradient_accumulation_step > 1:
                    loss = loss / self.args.gradient_accumulation_step

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                dev_loss += loss.item()
                nb_dev_steps += 1

        return dev_loss / nb_dev_steps

    def test_epoch(self, test_dataloader: DataLoader):
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


    def test_score_model(self, test_dataloader: DataLoader, use_zero=False):
        preds, y_test = self.test_epoch(test_dataloader)
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
        #print(len(y_test), len(preds))
        try:
            roc_auc_scores = roc_auc_score(y_test, preds)
        except ValueError:
            roc_auc_scores = 0.75

        return acc, mae, corr, f_score, roc_auc_scores

    def train(self, train_dataloader, validation_dataloader=None):
        # Enforce same train and dev batch size for ARL models
        if self.model_type == "mag-bert-arl":
            assert self.args.train_batch_size == self.args.dev_batch_size

        # Validation dataset must be present if early stopping is enabled
        if self.args.use_early_stopping:
            assert validation_dataloader is not None

        optimizer, scheduler = self.get_learner_optimizer()
        adv_optimizer, adv_scheduler = self.get_adversary_optimizer()
        global_step, epochs_trained, steps_trained_in_current_epoch = self.load_last_checkpoint(len(train_dataloader))
        patience = 0
        best_valid_loss = float('inf')

        for epoch_i in range(int(self.args.n_epochs)):
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_loss, global_step = self.train_epoch(train_dataloader, optimizer, scheduler, global_step, epoch_i, steps_trained_in_current_epoch, adv_optimizer, adv_scheduler)

            if validation_dataloader is not None:
                valid_loss = self.eval_epoch(validation_dataloader)
                if self.args.use_early_stopping:
                    if valid_loss < best_valid_loss:
                        patience = 0
                        best_valid_loss = valid_loss
                    else:
                        patience += 1
            else:
                valid_loss = float('inf')

            logging.info(
                "epoch:{}, train_loss:{}, valid_loss:{}".format(
                    epoch_i, train_loss, valid_loss
                )
            )

            if self.args.use_early_stopping and patience > self.args.early_stopping_patience:
                logging.info(f"Patience of {self.args.early_stopping_patience} steps reached.")
                logging.info("Training terminated.")
                break

        if not self.args.no_save:
            self.save_model(self.args.output_dir, optimizer, scheduler, adv_optimizer, adv_scheduler)

    def save_model(self, output_dir=None, optimizer=None, scheduler=None, adv_optimizer=None, adv_scheduler=None):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Saving model into {output_dir}")

        # Take care of distributed/parallel training
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        # Save model
        os.makedirs(os.path.join(output_dir), exist_ok=True)
        model_to_save.save_pretrained(output_dir)

        # Save training args
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        
        # Save optimizers and schedulers
        if optimizer and scheduler:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        if adv_optimizer and adv_scheduler:
            torch.save(adv_optimizer.state_dict(), os.path.join(output_dir, "optimizer-adv.pt"))
            torch.save(adv_scheduler.state_dict(), os.path.join(output_dir, "scheduler-adv.pt"))
