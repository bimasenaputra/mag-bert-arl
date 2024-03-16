import os
import pickle
import numpy as np
import pandas as pd
import ast

def read_segmented_data(acoustic_dir, visual_dir):
    # Read csv files
    acoustic = pd.read_csv(acoustic_dir)
    visual = pd.read_csv(visual_dir)
    segmented_data = visual
    segmented_data['acoustic'] = acoustic['acoustic']
    # Convert str to list type
    segmented_data['acoustic'] = segmented_data['acoustic'].apply(preprocess_string)
    segmented_data['visual'] = segmented_data['visual'].apply(preprocess_string)
    return segmented_data

def read_annotation(dir):
    with open(dir, "rb") as handle:
        annotation = pickle.load(handle, encoding='latin1')
        
        # Create a dictionary to hold the labels for each video
        video_targets = {}

        for label, inner_dict in annotation.items():
            for video, target in inner_dict.items():
                if video not in video_targets:
                    video_targets[video] = {}
                video_targets[video][label] = target

        # Convert the video_targets dictionary to a list of tuples
        converted_annotation = [(video.removesuffix('.mp4'), targets) for video, targets in video_targets.items()]

    return converted_annotation
    
# Convert values to list
def preprocess_string(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, AttributeError):
        return None

def split_visual_column(segmented_data):
    gaze = []
    facial_landmarks = []
    action_unit = []
    # Replace problematic rows with the list from the previous or next row
    for i, value in enumerate(segmented_data['visual']):
        try:
            gaze.append(np.array(value[:8+56*5], dtype=np.float32))
            facial_landmarks.append(np.array(value[8+56*5:8+56*5+6+68*5+40], dtype=np.float32))
            action_unit.append(np.array(value[8+56*5+6+68*5+40:], dtype=np.float32))
        except:
            gaze.append(None)
            facial_landmarks.append(None)
            action_unit.append(None)
    return gaze, facial_landmarks, action_unit

# Replace NaNs with element-wise average in a numpy array
def replace_nans_with_avg(row):
    masked_row = []
    for i in range(len(row)):
        if type(row[i]) is np.ndarray:
            masked_row.append(row[i])
            
    masked_row = np.array(masked_row, dtype=np.float32)
    average = np.average(masked_row, axis=0)
    
    # Replace NaNs with corresponding averages
    for i in range(len(row)):
        if type(row[i]) is float:
            row[i] = average
                
    return row
    
def remove_nans(column_list, column):
    column_np = column_list.reset_index()
    column_np[column] = column_np[column].apply(lambda x: replace_nans_with_avg(x))
    column_list = column_np.groupby('VideoID')[column].apply(list)

    return column_list

def process_data(annotation, segmented_data):
    segmented_data['VideoID'] = segmented_data['SegmentedBaseName'].str.extract(r'(.*)_segment_\d+')
    
    segmented_data['acoustic_numpy'] = segmented_data['acoustic'].apply(lambda x: np.array(x, dtype=np.float32))
    acoustic_list = segmented_data.groupby('VideoID')['acoustic_numpy'].apply(list)
    acoustic_list = remove_nans(acoustic_list,'acoustic_numpy')

    segmented_data['visual_numpy'] = segmented_data['visual'].apply(lambda x: np.array(x, dtype=np.float32))
    visual_list = segmented_data.groupby('VideoID')['visual_numpy'].apply(list)
    visual_list = remove_nans(visual_list,'visual_numpy')
    
    word_list = segmented_data.groupby('VideoID')['Word'].apply(list)

    ac_list = [np.array(acoustic_list[vid]) for vid, _ in annotation]
    vis_list = [np.array(visual_list[vid]) for vid, _ in annotation]
    wd_list = [word_list[vid] for vid, _ in annotation]

    return ac_list, vis_list, wd_list 

def convert_to_mag_input(annotation, wd_list, vis_list, ac_list):
    processed_data = []
    for i, (vid, row) in enumerate(annotation):
        wd_data = wd_list[i]
        vis_data = vis_list[i]
        ac_data = ac_list[i]
        label_id = np.array([row.get("interview",0.0),row.get("extraversion",0.0),row.get("neuroticism",0.0),row.get("agreeableness",0.0),row.get("conscientiousness",0.0),row.get("openness",0.0)], dtype=np.float32)
        segment = vid
        processed_data.append(((wd_data, vis_data, ac_data), label_id, segment))
    return processed_data

if __name__ == "__main__":
    # Path
    acoustic_train_dir = "~/first-impression/txt-train/sample_row_with_acoustic_whisper.csv"
    acoustic_dev_dir = "~/first-impression/txt-dev/sample_row_with_acoustic_whisper.csv"
    acoustic_test_dir = "~/first-impression/txt-test/sample_row_with_acoustic_whisper.csv"
    visual_train_dir = "~/first-impression/txt-train/openface.csv"
    visual_dev_dir = "~/first-impression/txt-dev/openface.csv"
    visual_test_dir = "~/first-impression/txt-test/openface.csv"
    train_annot_dir = "~/first-impression/txt-train/annotation_training.pkl"
    dev_annot_dir = "~/first-impression/txt-dev/annotation_validation.pkl"
    test_annot_dir = "~/first-impression/txt-test/annotation_test.pkl"
    output_dir = "~/first-impression/fiv2.pkl"

    # Expand absolute path
    acoustic_train_dir = os.path.expanduser(acoustic_train_dir)
    acoustic_dev_dir = os.path.expanduser(acoustic_dev_dir)
    acoustic_test_dir = os.path.expanduser(acoustic_test_dir)
    visual_train_dir = os.path.expanduser(visual_train_dir)
    visual_dev_dir = os.path.expanduser(visual_dev_dir)
    visual_test_dir = os.path.expanduser(visual_test_dir)
    train_annot_dir = os.path.expanduser(train_annot_dir)
    dev_annot_dir = os.path.expanduser(dev_annot_dir)
    test_annot_dir = os.path.expanduser(test_annot_dir)
    output_dir = os.path.expanduser(output_dir)

    # Read segmented_data
    print("...Reading segmented data...")
    segmented_data_train = read_segmented_data(acoustic_train_dir, visual_train_dir)
    segmented_data_dev = read_segmented_data(acoustic_dev_dir, visual_dev_dir)
    segmented_data_test = read_segmented_data(acoustic_test_dir, visual_test_dir)
    segmented_data = pd.concat([segmented_data_train, segmented_data_dev, segmented_data_test], ignore_index=True)

    # Read annotation
    print("...Reading annotations...")
    train_annotation = read_annotation(train_annot_dir)
    dev_annotation = read_annotation(dev_annot_dir)
    test_annotation = read_annotation(test_annot_dir)
    
    # Get modalities
    print("...Getting modalities...")
    ac_list_train, vis_list_train, wd_list_train = process_data(train_annotation, segmented_data)
    ac_list_dev, vis_list_dev, wd_list_dev = process_data(dev_annotation, segmented_data)
    ac_list_test, vis_list_test, wd_list_test = process_data(test_annotation, segmented_data)

    # Create inputs for MAG-BERT
    print("...Creating inputs...")
    processed_train_data = convert_to_mag_input(train_annotation, wd_list_train, vis_list_train, ac_list_train)
    processed_dev_data = convert_to_mag_input(dev_annotation, wd_list_dev, vis_list_dev, ac_list_dev)
    processed_test_data = convert_to_mag_input(test_annotation, wd_list_test, vis_list_test, ac_list_test)

    processed_data = {"train": processed_train_data, "dev": processed_dev_data, "test": processed_test_data}
    with open(output_dir, 'wb') as f:
        pickle.dump(processed_data, f)
