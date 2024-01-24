import os
import cv2
import opensmile
import detectron2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from pyAudioAnalysis import MidTermFeatures as aF

class FeatureExtractor(object):
    def __init__(self, video_path, audio_path, aligments):
        self.visual_extractor = VisualExtractor(video_path, aligments)
        self.acoustic_extractor = AcousticExtractor(audio_path, aligments)
        self.text_extractor = TextExtractor(aligments)

class AcousticExtractor(object):
    def __init__(self, path, aligments):
    	self.path = path
        self.aligment = aligment

    def egemaps(self):
        # todo
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        features_list = []
        path_list = [f.path for f in os.scandir(self.path) if f.is_dir()]

        for path in path_list:
            audio_files = [f.path for f in os.scandir(path) if f.is_file()]
            for filename in audio_files:
                acoustic_features = smile.process_files(
                    filename
                )    
                features_list.append(features_list.append(acoustic_features.iloc[0].values.tolist()))
        features_array = np.array(features_list)
        return features_array

    def pyaudioanalysis(self):
        path_list = [f.path for f in os.scandir(self.path) if f.is_dir()]
        features = aF.multiple_directory_feature_extraction(path_list, , 1, 1, 0.02, 0.02, compute_beat=False) 
        return features

class VisualExtractor(object):
    def __init__(self, path, aligments):
        self.path = path
        self.aligment = aligment

    def cnn_lstm(self):
        # init
        path_list = [f.path for f in os.scandir(self.path) if f.is_dir()]

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
            video_files = [f.path for f in os.scandir(path) if f.is_file()]
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
                          # filter the outputs with a good confidence score
                          persons, pIndicies = filter_persons(outputs)
                          if len(persons) >= 1:
                              # pick only pose estimation results of the first person.
                              # actually, we expect only one person to be present in the video. 
                              p = persons[0]
                              # input feature array for lstm
                              features = []
                              # add pose estimate results to the feature array
                              for i, row in enumerate(p):
                                  features.append(row[0])
                                  features.append(row[1])

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
    def __init__(self, aligments):
        self.aligment

    def words(self):
        words = []

        for alignment in aligments:
            words.append([word_aligment["word"] for word_aligment in aligment])

        return words 