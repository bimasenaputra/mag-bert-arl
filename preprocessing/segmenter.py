import os
import spacy

from pydub import AudioSegment
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

nlp = spacy.load("en_core_web_sm")

def segment_transcription(transcription):
	doc = nlp(transcription)
	sentences = [sent.text for sent in doc.sents]
	return sentences

def segment_sentence(sentence):
	doc = nlp(transcription)
	tokens = [token for token in doc if not token.is_stop]
	return tokens

def get_segments(alignment, transcription):
	sentences = segment_transcription(transcription)
	segments_time_windows = []
	segment_alignments = []

	l, r = 0, 1

	tokenized_sentences = []
	tokenized_sentences_length = 0

	for sentence in sentences:
		tokens = segment_sentence(sentence)
		assert len(tokens) > 1
		tokenized_sentences_length += len(tokens)
		tokenized_sentences.append((tokens[0], tokens[-1]))

	assert tokenized_sentences_length == len(alignment)

	for (first_token, last_token) in tokenized_sentences:
		segment_alignment = []

		while l < len(alignment) and alignment[l]["word"] != first_token:
			l += 1

		begin = alignment[l]
		begin_time = begin["begin"]
		segment_alignment.append(dict(word=first_token, begin=0, end=begin["end"] - begin_time))

		r = l+1
		while r < len(alignment) and alignment[r]["word"] != last_token:
			current = alignment[r]
			segment_alignment.append(dict(word=current["word"], begin=current["begin"] - begin_time, end=current["end"] - begin_time))
			r += 1

		end = alignment[r]
		end_time = end["end"]
		segment_alignment.append(dict(word=last_token, begin=end["begin"] - begin_time, end=end_time - begin_time))

		segments_time_windows.append((begin, end))
		segment_alignments.append(segment_alignment)

		l = r+1

	return segments_time_windows, segment_alignments

def segment_audio_file(audio_file_path, segments_time_windows, output):
    audio = AudioSegment.from_file(audio_file_path)
    filename = os.path.basename(audio_file_path)
    output_file_path = "{path}/{name}".format(path=output, name=filename.removesuffix('.wav'))

    for i, (begin, end) in enumerate(segments_time_windows):
    	audio_segment = audio[begin:end]
    	audio_segment.export("{name}[{no}].wav".format(name=output_file_path, no=i), format="wav")

def segment_video_file(video_file_path, segments_time_windows, output):
    filename = os.path.basename(video_file_path)
    output_file_path = "{path}/{name}".format(path=output, name=filename.removesuffix('.mp4'))

    for i, (begin, end) in enumerate(segments_time_windows):
    	ffmpeg_extract_subclip(video_file_path, begin, end, targetname="{name}[{no}].mp4".format(name=output_file_path, no=i))

def segment_video_audio_files(video_folder, audio_folder, alignments, transcriptions, output_video=None, output_audio=None):
	if output_video is None:
		output = video_folder

	if output_audio is None:
		output = audio_folder

	segment_alignments = []
	for i, (video_filename, audio_filename) in enumerate(zip(video_folder, audio_folder)):
		segments_time_windows, segment_alignments_tmp = get_segments(alignments[i], transcriptions[i])
		if video_filename.endswith('.mp4'):
			video_file_path = os.path.join(video_folder, video_filename)
			segment_video_file(video_file_path, segments_time_windows, output)
        if audio_filename.endswith('.wav'):
        	audio_file_path = os.path.join(audio_folder, audio_filename)
            segment_audio_file(audio_file_path, segments_time_windows, output)
        segment_alignments.extend(segment_alignments_tmp)

     return segment_alignments