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

	i, l, r = 0, 0, 1

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

		assert alignment[l]["word"] == first_token

		segment_alignment.append(dict(word=first_token, begin=0, end=alignment[l]["end"] - alignment[l]["begin"]))
		segments_time_windows.append((i, alignment[l]["begin"], alignment[l]["end"]))

		r = l+1
		while r < len(alignment) and alignment[r]["word"] != last_token:
			segment_alignment.append(dict(word=alignment[r]["word"], begin=0, end=alignment[r]["end"] - alignment[r]["begin"]))
			segments_time_windows.append((i, alignment[r]["begin"], alignment[r]["end"]))
			r += 1

		segment_alignment.append(dict(word=last_token, begin=0, end=alignment[r]["end"] - alignment[r]["begin"]))
		segments_time_windows.append((i, alignment[r]["begin"], alignment[r]["end"]))


		segment_alignments.append(segment_alignment)

		l = r+1
		i += 1

	return segments_time_windows, segment_alignments

def segment_audio_file(audio_file_path, segments_time_windows, output):
    audio = AudioSegment.from_file(audio_file_path)
    filename = os.path.basename(audio_file_path)
    output_file_path = "{path}/{name}".format(path=output, name=filename.removesuffix('.wav'))
    idx = 0

    for (i, begin, end) in enumerate(segments_time_windows):
    	audio_segment = audio[begin:end]
    	audio_segment.export("{name}[{no}]/{name}[{no}][{idx}].wav".format(name=output_file_path, no=i, idx=idx), format="wav")
    	idx += 1

def segment_video_file(video_file_path, segments_time_windows, output):
    filename = os.path.basename(video_file_path)
    output_file_path = "{path}/{name}".format(path=output, name=filename.removesuffix('.mp4'))
    idx = 0

    for (i, begin, end) in enumerate(segments_time_windows):
    	ffmpeg_extract_subclip(video_file_path, begin, end, targetname="{name}[{no}]/{name}[{no}][{idx}].mp4".format(name=output_file_path, no=i, idx=idx))
    	idx += 1

def segment_video_audio_files(video_folder, audio_folder, alignments, transcriptions, output_video=None, output_audio=None):
	if output_video is None:
		output_video = video_folder

	if output_audio is None:
		output_audio = audio_folder

	segment_alignments = []
	for i, (video_filename, audio_filename) in enumerate(zip(video_folder, audio_folder)):
		segments_time_windows, segment_alignments_tmp = get_segments(alignments[i], transcriptions[i])
		if video_filename.endswith('.mp4'):
			segment_video_file(video_filename, segments_time_windows, output_video)
        if audio_filename.endswith('.wav'):
            segment_audio_file(audio_filename, segments_time_windows, output_audio)
        segment_alignments.extend(segment_alignments_tmp)

     return segment_alignments