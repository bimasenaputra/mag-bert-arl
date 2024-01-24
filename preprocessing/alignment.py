import string
import whisper

from IPython.display import display, HTML
from whisper.tokenizer import get_tokenizer
from dtw import dtw
from scipy.ndimage import median_filter

AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

class WhisperAlignment(object):
	def __init__(self, model, language):
		self.medfilt_width = 7
		self.qk_scale = 1.0
		self.model = model

		self.tokenizer = get_tokenizer(model.is_multilingual, language=language)
		if language in {"chinese", "japanese", "thai", "lao", "myanmar"}:
			self.split_tokens = self.split_tokens_on_unicode
		else:
			self.split_tokens = self.split_tokens_on_spaces

	def split_tokens_on_unicode(self, tokens: torch.Tensor):
		words = []
		word_tokens = []
		current_tokens = []

		for token in tokens.tolist():
			current_tokens.append(token)
			decoded = self.tokenizer.decode_with_timestamps(current_tokens)
			if "\ufffd" not in decoded:
				words.append(decoded)
				word_tokens.append(current_tokens)
				current_tokens = []

		return words, word_tokens

	def split_tokens_on_spaces(self, tokens: torch.Tensor):
		subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
		words = []
		word_tokens = []

		for subword, subword_tokens in zip(subwords, subword_tokens_list):
			special = subword_tokens[0] >= self.tokenizer.eot
			with_space = subword.startswith(" ")
			punctuation = subword.strip() in string.punctuation
			if special or with_space or punctuation:
				words.append(subword)
				word_tokens.append(subword_tokens)
			else:
				words[-1] = words[-1] + subword
				word_tokens[-1].extend(subword_tokens)

		return words, word_tokens

	# return list of list of dicts, meaning list of audio's word alignment where each dict denotes word alignment in ms
	def get_alignment(self, audios, transcriptions):
		# install hooks on the cross attention layers to retrieve the attention weights
		QKs = [None] * self.model.dims.n_text_layer

		for i, block in enumerate(self.model.decoder.blocks):
			block.cross_attn.register_forward_hook(
				lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1])
			)

		result = []

		for audio, transcription in zip(audios, transcriptions):
			duration = len(audio)
			mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio)).cuda()
			tokens = torch.tensor(
				[
					*self.tokenizer.sot_sequence,
					self.tokenizer.timestamp_begin,
				] + self.tokenizer.encode(transcription) + [
					self.tokenizer.timestamp_begin + duration // AUDIO_SAMPLES_PER_TOKEN,
					self.tokenizer.eot,
				]
			).cuda()

			with torch.no_grad():
				logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))

			weights = torch.cat(QKs)  # layers * heads * tokens * frames    
			weights = weights[:, :, :, : duration // AUDIO_SAMPLES_PER_TOKEN].cpu()
			weights = median_filter(weights, (1, 1, 1, self.medfilt_width))
			weights = torch.tensor(weights * self.qk_scale).softmax(dim=-1)

			w = weights / weights.norm(dim=-2, keepdim=True)
			matrix = w[-6:].mean(axis=(0, 1))

			alignment = dtw(-matrix.double().numpy())

			jumps = np.pad(np.diff(alignment.index1s), (1, 0), constant_values=1).astype(bool)
			jump_times = alignment.index2s[jumps] * (1000 / AUDIO_TIME_PER_TOKEN)
			words, word_tokens = self.split_tokens(tokens)

			word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
			begin_times = jump_times[word_boundaries[:-1]]
			end_times = jump_times[word_boundaries[1:]]

			data = [
				dict(word=word, begin=begin, end=end)
				for word, begin, end in zip(words[:-1], begin_times, end_times)
				if not word.startswith("<|") and word.strip() not in ".,!?、。"
			]

			result.append(data)

		return result
