import os
import cv2
import torch
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up model and tokenizer
model_name = "rizvandwiki/gender-classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to predict gender from text
def predict_gender(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_gender = "Male" if outputs.logits.argmax().item() == 0 else "Female"
    return predicted_gender

# Function to process video and predict gender
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frames = random.sample(range(0, frame_count), min(3, frame_count))

    frames_data = []
    for frame_index in random_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            # Convert frame to text (e.g., using OCR)
            text_from_frame = "Text extracted from frame"  # Replace with your actual text extraction method
            predicted_gender = predict_gender(text_from_frame)
            frames_data.append(predicted_gender)

    cap.release()

    # Determine final label using voting
    if frames_data:
        final_label = max(set(frames_data), key=frames_data.count)
    else:
        final_label = "Unknown"

    return final_label

# Function to process all videos in directory
def process_videos_in_directory(video_source_dir):
    video_files = [file for file in os.listdir(video_source_dir) if file.endswith(".mp4")]
    for video_file in video_files:
        video_path = os.path.join(video_source_dir, video_file)
        predicted_gender = process_video(video_path)
        output_csv = f"{video_file.split('.')[0]}_output.csv"
        df = pd.DataFrame({"Video_File": [video_file], "Predicted_Gender": [predicted_gender]})
        df.to_csv(output_csv, index=False)

# Main function
if __name__ == "__main__":
    video_source_dir = "~/sakura_science_intern_dataset/videos/"
    video_source_dir = os.path.expanduser(video_source_dir)
    process_videos_in_directory(video_source_dir)
