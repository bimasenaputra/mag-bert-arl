import os
import opensmile
import pandas as pd
from tqdm import tqdm  # Import tqdm
import librosa

# Define the paths
output_dir_txt = '~/first-impression/txt-test/'
audio_source_dir = '~/first-impression/audio-test-seg/'

# Expand ~ to the absolute path of the user's home directory
output_dir_txt = os.path.expanduser(output_dir_txt)
audio_source_dir = os.path.expanduser(audio_source_dir)

# Load the segmented_words.csv file
csv_filepath = os.path.join(output_dir_txt, 'segmented_words_whisper.csv')
df_segment_info = pd.read_csv(csv_filepath)

# Extract the first row as a sample
sample_row = df_segment_info.copy()
# Initialize opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# Create a list to store the extracted features
features_list = []

# Use tqdm to add a progress bar
for _, row in tqdm(sample_row.iterrows(), total=len(sample_row), desc="Processing rows"):
    # Construct the full path to the segmented audio file
    audio_filepath = os.path.join(audio_source_dir, row['SegmentedBaseName'] + '.wav')

    # Extract features using opensmile
    features = smile.process_file(audio_filepath)

    # Convert features to a row list
    row_list = [audio_filepath] + features.values.tolist()

    row_list = row_list[1]
    features_list.append(row_list)

# Assign the entire features_list to the 'acoustic' column using .loc
sample_row.loc[:, 'acoustic'] = features_list

# Save the modified DataFrame to a CSV file in the output directory
output_csv_filepath = os.path.join(output_dir_txt, 'sample_row_with_acoustic_whisper.csv')
sample_row.to_csv(output_csv_filepath, index=False)

print(f"DataFrame saved to: {output_csv_filepath}")
