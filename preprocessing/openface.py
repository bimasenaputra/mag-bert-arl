import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Define the paths
output_dir_txt = '~/first-impression/txt-train/'
video_source_dir = '~/first-impression/openface/features-train'

# Expand ~ to the absolute path of the user's home directory
output_dir_txt = os.path.expanduser(output_dir_txt)
video_source_dir = os.path.expanduser(video_source_dir)

# Load the segmented_words.csv file
csv_filepath = os.path.join(output_dir_txt, 'segmented_words_whisper.csv')
df = pd.read_csv(csv_filepath)

buffer_milliseconds = 0.333
features_list = []

# Read all necessary CSV files once
csv_files = os.listdir(video_source_dir)
data = {}
for csv_file in tqdm(csv_files, desc='Reading CSV files'):
    if csv_file.endswith(".csv"):
        key = csv_file.removesuffix(".csv")
        if key not in data:
            data[key] = pd.read_csv(os.path.join(video_source_dir, csv_file))
        else:
            data[key] = pd.concat([data[key], pd.read_csv(os.path.join(video_source_dir, csv_file))], ignore_index=True)

# Process DataFrame
for index, row in tqdm(df.iterrows(), total=len(df), desc='Processing rows'):
    start_val = row['Start']
    end_val = row['End']
    begin_adjusted = start_val - buffer_milliseconds/2
    end_adjusted = end_val + buffer_milliseconds/2
    input_string = row["SegmentedBaseName"]
    key = input_string.split(".")[0]
    key += input_string[len(key):len(key)+4]
    openface = data[key]
    mask = (openface['frame'] * buffer_milliseconds > begin_adjusted) & (openface['frame'] * buffer_milliseconds < end_adjusted)
    mean_values = openface[mask].iloc[:, 5:].mean(axis=0)
    features_list.append(mean_values.tolist())

df['visual'] = features_list

# Save the modified DataFrame to a CSV file in the output directory
output_csv_filepath = os.path.join(output_dir_txt, 'openface.csv')
df.to_csv(output_csv_filepath, index=False)

print(f"DataFrame saved to: {output_csv_filepath}")
