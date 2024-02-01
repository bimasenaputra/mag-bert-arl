import os
import pandas as pd
import tqdm
import subprocess

def segment_audio_file(audio_file_path, segments_time_windows, output):
	_, filename = os.path.split(audio_file_path)
	filename = filename.removesuffix(".wav")
	idx = 0
	last_i = 0

	for (i, begin, end) in segments_time_windows:
		output_file_path = "{path}{name}[{no}]".format(path=output, name=filename, no=i)
		if not os.path.exists(output_file_path):
			os.makedirs(output_file_path)
		output_file = "{output}/{name}[{no}][{idx}].wav".format(output=output_file_path, name=filename, no=i, idx=idx)
		subprocess_cmd = 'ffmpeg -loglevel quiet -stats -c:a copy -nostdin -y -ss {start}ms -to {end}ms -i {inp} {outp}'.format(inp=audio_file_path, outp=output_file, start=begin, end=end).split()
		subprocess.run(subprocess_cmd)
		#subprocess.run(["echo","$$"])
		if i == last_i:
			idx += 1
		else:
			idx = 0

def segment_video_file(video_file_path, segments_time_windows, output):
	_, filename = os.path.split(video_file_path)
	filename = filename.removesuffix(".mp4")
	idx = 0
	last_i = 0
	
	for (i, begin, end) in segments_time_windows:
		output_file_path = "{path}{name}[{no}]".format(path=output, name=filename, no=i)
		if not os.path.exists(output_file_path):
			os.makedirs(output_file_path)
		output_file = "{output}/{name}[{no}][{idx}].mp4".format(output=output_file_path, name=filename, no=i, idx=idx)
		subprocess_cmd = 'ffmpeg -loglevel quiet -stats -c copy -nostdin -y -ss {start}ms -to {end}ms -i {inp} {outp}'.format(inp=video_file_path, outp=output_file, start=begin, end=end).split()
		subprocess.run(subprocess_cmd)
		#subprocess.run(["echo","$$"])
		if i == last_i:
			idx += 1
		else:
			idx = 0

def segment_video_audio_files(video_folder, audio_folder, segment_time_windows, output_video=None, output_audio=None):
	if output_video is None:
		output_video = video_folder

	if output_audio is None:
		output_audio = audio_folder

	for i, (video_filename, audio_filename) in enumerate(tqdm.tqdm(zip(video_folder, audio_folder), total=len(video_folder))):
		if video_filename.endswith('.mp4'):
			segment_video_file(video_filename, segment_time_windows[i], output_video)
		if audio_filename.endswith('.wav'):
			segment_audio_file(audio_filename, segment_time_windows[i], output_audio)


# return a list of vector for each segmented file
def segment_csv_file(audio_file, segments_time_windows, csv_dir, buffer_milliseconds = 0.333, columns_to_drop = 5):

	df = pd.read_csv(csv_dir + "/" + audio_file + ".csv")
	
	for (_, begin, end) in segments_time_windows:
		begin_adjusted = begin - buffer_milliseconds/2
		end_adjusted = end + buffer_milliseconds/2
		# Calculate the mean of all rows in the filtered DataFrame
		mask = (df['frame'] * buffer_milliseconds > begin_adjusted) & (df['frame'] * buffer_milliseconds < end_adjusted)
		mean_values = df[mask].mean(axis=0)
		# Convert the mean values to a list
		mean_list = mean_values.tolist()
		# Drop the first 5 elements from the list
		mean_list = mean_list[:columns_to_drop]

	return mean_list
