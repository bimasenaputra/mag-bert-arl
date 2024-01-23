import os
import opensmile

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

        for i, aligment in enumerate(alignments):
            audio_filename = audios.files[i]

            acoustic_features = smile.process_files(
                audio.files,
                root=audio.root,
            )    

    def pyaudioanalysis(self):
        path_list = [f.path for f in os.scandir(self.path) if f.is_dir()]
        return aF.multiple_directory_feature_extraction(path_list, , 1, 1, 0.02, 0.02, compute_beat=False)

class VisualExtractor(object):
    def __init__(self, path, aligments):
        self.path = path
        self.aligment = aligment

    def cnn_lstm(self):
        pass

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