import os
import multiprocessing
import cv2
import opensmile
import detectron2
import numpy as np
import torch.nn as nn
import torch
import pickle as pkl

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from pyAudioAnalysis import MidTermFeatures as aF

class FeatureExtractor(object):
    def __init__(self, video_path, audio_path, alignments):
        self.path = video_path
        self.visual_extractor = VisualExtractor(video_path)
        self.acoustic_extractor = AcousticExtractor(audio_path)
        self.text_extractor = TextExtractor(alignments)

    def extract(self, visual="cnn_lstm", acoustic="pyaudioanalysis", text="words"):
        visual_features = None
        acoustic_features = None
        text_features = None

        # visual
        if visual == "cnn_lstmi":
            visual_features = self.visual_extractor.cnn_lstm()
        elif visual == "open_face":
            visual_features = self.visual_extractor.open_face()

        # acoustic
        if acoustic == "pyaudioanalysis":
            acoustic_features = self.acoustic_extractor.pyaudioanalysis()
        elif acoustic == "egemaps":
            acoustic_features = self.acoustic_extractor.egemaps()

        # text
        if text == "words":
            text_features = self.text_extractor.words()

        # segments
        segments_dir = [f.path for f in sorted(os.scandir(self.path), key=lambda x: x.name) if f.is_dir()]
        segments = []

        for dirname in segments_dir:
            _, base_name = os.path.split(dirname)
            files = [f.name for f in os.scandir(dirname) if f.is_file()]

            for idx, filename in enumerate(files):
                segments_name = f"{base_name}[{idx}]"
                segments.append(segments_name)
        print(len(segments))
        assert len(text_features) == len(segments)
        #assert len(visual_features) == len(acoustic_features)
        #assert len(visual_features) == len(text_features)

        features = [(text_features, acoustic_features, visual_features), segments]
        return features

class AcousticExtractor(object):
    def __init__(self, path):
        self.path = path
        
    def egemaps(self):
        # todo
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
            num_workers=multiprocessing.Pool()._processes
        )
        features_list = []
        path_list = [f.path for f in sorted(os.scandir(self.path), key=lambda x: x.name) if f.is_dir()]

        for path in path_list:
            audio_files = [f.path for f in sorted(os.scandir(path), key=lambda x: x.name) if f.is_file()]
            for filename in audio_files:
                acoustic_features = smile.process_files(
                    filename
                )    
                features_list.append(acoustic_features.iloc[0].values.tolist)
        features_array = np.array(features_list)
        return features_array

    def pyaudioanalysis(self):
        path_list = [f.path for f in sorted(os.scandir(self.path), key=lambda x: x.name) if f.is_dir()]
        features = aF.multiple_directory_feature_extraction(path_list, 1, 1, 0.02, 0.02, compute_beat=False) 
        with open("pyaudio.pkl", "wb") as f:
                pkl.dump(features, f)
        return features

class VisualExtractor(object):
    def __init__(self, path):
        self.path = path

    def cnn_lstm(self):
        # init
        path_list = [f.path for f in sorted(os.scandir(self.path), key=lambda x: x.name) if f.is_dir()]

        cfg = get_cfg()
        SKIP_FRAME_COUNT = 0
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        pose_detector = DefaultPredictor(cfg)

        lstm = nn.LSTM(1000, 50, batch_first=True)

        features_list = []

        for path in path_list:
            segment_features_list = []
            video_files = [f.path for f in sorted(os.scandir(path), key=lambda x: x.name) if f.is_file()]
            for filename in video_files:
                if filename.endswith('.mp4'):
                    averaged_features_list = []
                    # open the video
                    cap = cv2.VideoCapture(filename)                    
                    # counter
                    counter = 0
                    # buffer to keep the output of detectron2 pose estimation
                    buffer_window = []

                    while True:
                        # read the frame
                        ret, frame = cap.read()
                        # return if end of the video
                        if ret == False:
                            break
                        if(counter % (SKIP_FRAME_COUNT+1) == 0):             
                          # predict pose estimation on the frame
                          outputs = pose_detector(frame)          
                          instances = outputs['instances']
                          pred_boxes = instances.pred_boxes.tensor
                          scores = instances.scores

                          # Filter persons based on confidence threshold
                          persons = [pred_boxes[i] for i in range(len(scores)) if scores[i] > 0.95]
                          pIndices = [i for i in range(len(scores)) if scores[i] > 0.95]
                          if len(persons) >= 1:
                              # pick only pose estimation results of the first person.
                              # actually, we expect only one person to be present in the video. 
                              p = persons[0]
                              # input feature array for lstm
                              features = []
                              # add pose estimate results to the feature array
                              for i, row in enumerate(p):
                                  print(row)
                                  features.append(row[0].item())
                                  features.append(row[1].item())

                              # append the feature array into the buffer
                              # not that max buffer size is 32 and buffer_window operates in a sliding window fashion
                              if len(buffer_window) < WINDOW_SIZE:
                                  buffer_window.append(features)
                              else:
                                  # convert input to tensor
                                  model_input = torch.Tensor(np.array(buffer_window, dtype=np.float32))
                                  # add extra dimension
                                  model_input = torch.unsqueeze(model_input, dim=0)
                                  # extract the features using lstm
                                  extracted_features = lstm(model_input)

                                  # Append the entire buffer to the list
                                  averaged_features_list.append(np.array(extracted_features))

                                  # pop the first value from buffer_window and add the new entry in FIFO fashion, to have a sliding window of size 32.
                                  buffer_window.pop(0)
                                  buffer_window.append(features)

                        counter += 1
                        cap.release()

                    # Calculate the average of features outside the loop
                    averaged_features_array = np.mean(averaged_features_list, axis=0)
                    segment_features_list.append(averaged_features_array)

            segment_features_array = np.array(token_features_list)
            features_list.append(segment_features_array)
        # Convert the list of averaged features to a numpy array
        features_array = np.array(features_list)

        return features_array

    def open_face(self):
        # todo
        pass

class TextExtractor(object):
    def __init__(self, alignments):
        self.alignments = alignments

    def words(self):
        words = []

        for alignment in self.alignments:
            words.append([word_alignment["text"] for word_alignment in alignment])
        with open("words.pkl", "wb") as f:
                pkl.dump(words, f)
        print(words[:2])
        return words 
