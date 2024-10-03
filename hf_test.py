import pickle
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.io import wavfile
import whisper_timestamped as whisper


from load_dataset import get_appropriate_dataset
from argparse_utils import str2bool, seed, set_random_seed
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE, BERT_PRETRAINED_MODEL_ARCHIVE_LIST
from feature_extractor_mag import MAGBertFeatureExtractor
from processing_mag import MAGBERTProcessor
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum number of tokens the model can take in a single input")
parser.add_argument("--num_labels", type=int, default=1, help="Number of classes/labels to predict")
parser.add_argument("--model", type=str, default="mag-bert-base-uncased-4-tv", help="Name of model to test")
parser.add_argument("--tokenizer", type=str, choices=BERT_PRETRAINED_MODEL_ARCHIVE_LIST, default="bert-base-uncased", help="Bert tokenizer to use")
parser.add_argument("--beta_shift", type=float, default=1.0, help="The constant 'beta' to be used in the adaption gate during feature fusion with other features")
parser.add_argument("--dropout_prob", type=float, default=0.5, help="Probability of a neuron being dropped out during each training session")
parser.add_argument("--filepath", type=str, default="audio.wav", help="Path to the audio file")
parser.add_argument("--cuda_device", type=int, default=0, help="Cuda device to use for training")
parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPU used for training")
args = parser.parse_args()
args.device = DEVICE

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob

def test_epoch(test_dataloader):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(args.device) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
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

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            preds.extend(logits)
            preds = np.array(preds)

    return preds, labels

def set_up_data_loader():
    # Init
    fe = MAGBertFeatureExtractor()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    proc = MAGBERTProcessor(fe, tokenizer)
    
    # Transcribe
    audio = whisper.load_audio(args.filepath)
    model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
    result = whisper.transcribe(model, audio, language="en")

    # Extract information from Whisper result
    words = []
    segments = []

    for segment in result['segments']:
        for word_data in segment['words']:
            words.append(word_data['text'])
            segments.append([word_data['start'], word_data['end']])

    # Preprocess
    samplerate, raw_speech = wavfile.read(args.filepath)
    data = proc(words, segments, raw_speech, max_seq_length=args.max_seq_length, sampling_rate=samplerate, acoustic_dim=ACOUSTIC_DIM, visual_dim=VISUAL_DIM, return_tensors='pt')

    return data

def main():
    test_dataloader = set_up_data_loader()
    if torch.cuda.is_available():
        if cuda_device == -1:
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{cuda_device}")
    else:
        device = "cpu"
        
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )

    model = MAG_BertWithARL.from_pretrained(args.model, multimodal_config=multimodal_config, num_labels=args.num_labels)
    model.to(device)
    if args.n_gpu > 1:
        model = DataParallel(model)

    preds = test_epoch(test_dataloader)
    print(preds)

if __name__ == "__main__":
    main()
