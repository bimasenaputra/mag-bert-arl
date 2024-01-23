import os
import multiprocessing
import spacy
import pickle as pkl

from fiv2 import AudioDataset
from alignment import WhisperAlignment
from segmenter import segment_video_audio_files
from FeatureExtractor import FeatureExtractor
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

video_folder = os.path.expanduser("~/video/")
audio_folder = os.path.expanduser("~/audio/")
video_segment_folder = os.path.expanduser("~/video-seg/")
audio_segment_folder = os.path.expanduser("~/audio-seg/")
whisper_model = whisper.load_model("base")
whisper_alignment = WhisperAlignment(whisper_model, "english")
audios = AudioDataset(audio_folder)

def get_transcriptions():
    transcriptions = []
    for filename in audios.files:
        if filename.endswith('.wav'):
            audio_file_path = os.path.join(audio_folder, filename)
            transcription = whisper_model.transcribe(audio_file_path, fp16=False)
            transcriptions.append(transcription)

    return transcriptions

if not os.path.exists('data.pkl'):
    os.mknod('data.pkl')

with open("data.pkl", "rb") as handle:
        data = pickle.load(handle)

transcriptions = get_transcriptions()
alignments = whisper_alignment.get_alignment(audios, transcriptions)
alignments = segment_video_audio_files(video.files, audio.files, alignments, transcriptions, video_segment_folder, audio_segment_folder)



with open('data.pkl', 'wb') as f:
    pkl.dump(data, f)