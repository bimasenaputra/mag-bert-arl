import os
import librosa
import torch
import torchaudio
import pandas as pd
import soundfile as sf
from itertools import groupby
from tqdm import tqdm
import whisper_timestamped as whisper

model = whisper.load_model("tiny", device="cuda")

# Directories
output_dir_audio = '~/first-impression/audio-train-seg/'
output_dir_video = '~/first-impression/video-train-seg/'
output_dir_txt = '~/first-impression/txt-train/'
audio_source_dir = '~/first-impression/audios-train/'
video_source_dir = '~/first-impression/videos-train/'

# Expand ~ to the absolute path of the user's home directory
output_dir_audio = os.path.expanduser(output_dir_audio)
output_dir_video = os.path.expanduser(output_dir_video)
output_dir_txt = os.path.expanduser(output_dir_txt)
audio_source_dir = os.path.expanduser(audio_source_dir)
video_source_dir = os.path.expanduser(video_source_dir)

# Create directories if they don't exist
os.makedirs(output_dir_audio, exist_ok=True)
os.makedirs(output_dir_video, exist_ok=True)
os.makedirs(output_dir_txt, exist_ok=True)

# Get a list of all audio files in the sample directory
audio_files = [f for f in os.listdir(audio_source_dir) if f.endswith('.wav')]

# Create lists to store information
segment_info_list = []
confidence_mean_list = []

# Iterate through each audio file
for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
    # Load the audio file
    audio_filepath = os.path.join(audio_source_dir, audio_file)
    # speech, original_sample_rate = librosa.load(audio_filepath, sr=None)
    # resampled_speech = librosa.resample(speech, orig_sr=original_sample_rate, target_sr=target_sample_rate)
    speech = whisper.load_audio(audio_filepath)
    # Transcribe audio using Whisper
    result = whisper.transcribe(model, speech, language="en")

    # Extract base filename
    base_filename = os.path.splitext(os.path.basename(audio_filepath))[0]

    # Extract information from Whisper result
    words = []
    word_start_times = []
    word_end_times = []
    confidences = []

    for segment in result['segments']:
        for word_data in segment['words']:
            words.append(word_data['text'])
            word_start_times.append(word_data['start'])
            word_end_times.append(word_data['end'])
            confidences.append(word_data['confidence'])

    # Calculate mean confidence
    mean_confidence = sum(confidences) / len(confidences)

    # Save confidence mean and audio filename
    confidence_mean_list.append({'AudioFilename': base_filename, 'MeanConfidence': mean_confidence})

    # Adjust start and end times as needed

    # Save segmented audio files and update the list
    for i, (word, start_time, end_time) in enumerate(zip(words, word_start_times, word_end_times)):
        start_sample = int(start_time * 16000)
        end_sample = int(end_time * 16000)

        audio_segment = speech[start_sample:end_sample]

        output_filepath = os.path.join(output_dir_audio, f'{base_filename}_segment_{i}.wav')
        sf.write(output_filepath, audio_segment, 16000)

        # Append information to the list
        segment_info_list.append({'SegmentedBaseName': f'{base_filename}_segment_{i}', 'Word': word, 'Start': start_time, 'End': end_time, 'Confidence': confidences[i]})

# Convert the lists to DataFrames
df_segment_info = pd.DataFrame(segment_info_list)
df_confidence_mean = pd.DataFrame(confidence_mean_list)

# Save DataFrames to CSV files
csv_filepath_segment_info = os.path.join(output_dir_txt, 'segmented_words_whisper.csv')
csv_filepath_confidence_mean = os.path.join(output_dir_txt, 'confidence_mean.csv')

df_segment_info.to_csv(csv_filepath_segment_info, index=False)
df_confidence_mean.to_csv(csv_filepath_confidence_mean, index=False)
