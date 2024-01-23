import os

class AudioDataset(object):
	def __init__(self, audio_folder):
        all_audio = []
        length = 0
        files = []

        for filename in sorted(os.listdir(audio_folder)):
        	if filename.endswith('.wav'):
                files.append(filename)
        		audio_file_path = os.path.join(audio_folder, filename)
            	with open(audio_file_path, 'rb') as file:
            		audio_bytes = file.read()
            		all_audio.append(wavfile.read(io.BytesIO(audio_bytes))[1])
            	length += 1

        self.all_audio = all_audio
        self.length = length
        self.root = audio_folder
        self.files = files

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        audio = torch.from_numpy(self.all_audio[idx].copy())
        return audio