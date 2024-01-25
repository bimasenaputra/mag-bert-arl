import os
import torch
import pickle as pkl
import whisper_timestamped as whisper

from segmenter import segment_video_audio_files
from feature_extractor import FeatureExtractor
"""
format dataset:
{
    train:　【
        klip video 1 yang berisi tuple fitur, label_ids (target), dan segment,
        klip video 2 sama isinya
    】
}

words (List[str]): List of words
visual (np.array): Numpy array of shape (sequence_len, VISUAL_DIM)
acoustic (np.array): Numpy array of shape (seqeunce_len, ACOUSTIC_DIM)
label_id (float / numpy.ndarray): Label for data point
segment (str): Unique identifier for each data point
"""

video_folder = os.path.expanduser("~/videos/")
audio_folder = os.path.expanduser("~/audios/")
video_segment_folder = os.path.expanduser("~/video-seg/")
audio_segment_folder = os.path.expanduser("~/audio-seg/")
annotation_file = os.path.expanduser("~/labels/")
audio_files = [f.path for f in sorted(os.scandir(video_folder), key=lambda x: x.name) if f.is_file()]
video_files = [f.path for f in sorted(os.scandir(audio_folder), key=lambda x: x.name) if f.is_file()]

device = "cuda" if torch.cuda.is_available() else "cpu"
language = "en"
whisper_model = whisper.load_model("NbAiLab/whisper-large-v2-nob", device=device)

def get_segments():
    segment_time_windows = []
    segment_alignments = []
    for filename in audios_files:
        if filename.endswith('.wav'):
            result = whisper.transcribe(whisper_model, filename, language=language)
            for i, segment in enumerate(sorted(result['segments'], key=lambda x: x['id'])):
                for word in segment["words"]:
                    segment_time_windows.append((i, word["start"], word["end"]))
                    # avoid precision error
                    segment_alignments.append(dict(text=word["text"], start=0, end=((1000*word["end"])-(1000*word["start"]))/1000))

    return segment_time_windows, segment_alignments

if not os.path.exists('data.pkl'):
    data = {}
    with open('data.pkl', 'wb') as f:
        pkl.dump(data, f)

with open("data.pkl", "rb") as handle:
    data = pkl.load(handle)

""" features """
segment_time_windows, segment_alignments = get_segments()
segment_video_audio_files(video_files, audio_files, segment_time_windows, video_segment_folder, audio_segment_folder)
feature_extractor = FeatureExtractor(video_segment_folder, audio_segment_folder, segment_alignments)
features = feature_extractor.extract()


""" labels """
with open(annotation_file, "rb") as handle:
    annotation = pkl.load(handle, encoding='latin1')["interview"]

segments_dir = [f.path for f in sorted(os.scandir(video_segment_folder), key=lambda x: x.name) if f.is_dir()]
labels = []

for dirname in segments_dir:
    base_name = os.path.basename(dirname).split("[")[0]
    key_name = base_name + ".mp4"
    labels.append(annotation[key_name])

assert len(labels) == len(features[1])

entry = features + [labels]
transposed_entry = [[row[i] for row in entry] for i in range(len(entry[0]))]

data["train"] = transposed_entry

with open('data.pkl', 'wb') as f:
    pkl.dump(data, f)