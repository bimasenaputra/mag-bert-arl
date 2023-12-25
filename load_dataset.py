import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_to_features(examples, max_seq_length, tokenizer, visual_dim, acoustic_dim):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_bert_input(
            tokens, visual, acoustic, tokenizer, max_seq_length, visual_dim, acoustic_dim
        )

        # Check input length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert acoustic.shape[0] == max_seq_length
        assert visual.shape[0] == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def prepare_bert_input(tokens, visual, acoustic, tokenizer, max_seq_length, visual_dim, acoustic_dim):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, acoustic_dim))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, visual_dim))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, acoustic_dim))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, visual_dim))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids

def get_appropriate_dataset(data, max_seq_length, model_tokenizer, visual_dim, acoustic_dim):

    tokenizer = BertTokenizer.from_pretrained(model_tokenizer)

    features = convert_to_features(data, max_seq_length, tokenizer, visual_dim, acoustic_dim)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor(
        [f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset