"""
Speech processor class for Wav2Vec2
"""

import warnings
from contextlib import contextmanager
import numpy as np
import torch

from transformers import AutoFeatureExtractor, AutoTokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding


class MAGBERTProcessor(ProcessorMixin):
    r"""
    Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single
    processor.

    [`Wav2Vec2Processor`] offers all the functionalities of [`Wav2Vec2FeatureExtractor`] and [`PreTrainedTokenizer`].
    See the docstring of [`~Wav2Vec2Processor.__call__`] and [`~Wav2Vec2Processor.decode`] for more information.

    Args:
        feature_extractor (`Wav2Vec2FeatureExtractor`):
            An instance of [`Wav2Vec2FeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except (OSError, ValueError):
            warnings.warn(
                f"Loading a tokenizer inside {cls.__name__} from a config that does not"
                " include a `tokenizer_class` attribute is deprecated and will be "
                "removed in v5. Please add `'tokenizer_class': 'Wav2Vec2CTCTokenizer'`"
                " attribute to either your `config.json` or `tokenizer_config.json` "
                "file to suppress this warning: ",
                FutureWarning,
            )

            feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

            return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(
        self, 
        word_list: Union[List[str], List[List[str]]], 
        segments: Optional[Union[List[List[float]], List[List[int]], List[List[str]], List[List[List[float]]], List[List[List[int]]], List[List[List[str]]]]] = None,
        raw_speech: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        visuals: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        label_ids: Optional[float, List[float]] = None,
        max_seq_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        acoustic_dim: Optional[int] = None,
        visual_dim: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        *args,
        **kwargs
    ) -> BatchEncoding:
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        [`~Wav2Vec2FeatureExtractor.__call__`] and returns its output. If used in the context
        [`~Wav2Vec2Processor.as_target_processor`] this method forwards all its arguments to PreTrainedTokenizer's
        [`~PreTrainedTokenizer.__call__`]. Please refer to the docstring of the above two methods for more information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        if max_seq_length is None:
            raise ValueError("max_seq_length is not given")

        if isinstance(word_list[0], str):
            word_list = [word_list]

        if label_ids is not None and isinstance(label_ids, float):
            label_ids = [label_ids]
        else:
            label_ids = [0]

        if raw_speech is None:
            if acoustic_dim is None:
                raise ValueError("Please provide acoustic_dim parameter, equals to MAG-BERT acoustic input size, if raw_speech is not given.")
            else:
                raw_speech = torch.zeros(len(label_ids), acoustic_dim)
        else:
            acoustics = self.feature_extractor(audio, segments, *args, sampling_rate=sampling_rate, **kwargs)['acoustic']
            acoustic_dim = self.feature_extractor.feature_size

        if visuals is None:
            if visual_dim is None:
                raise ValueError("Please provide visual_dim parameter, equals to MAG-BERT acoustic input size, if visuals is not given.")
            else:
                visuals = torch.zeros(len(label_ids), visual_dim)
        else:
            visuals_is_batched = isinstance(visuals, np.ndarray) and len(visuals.shape) > 1 or (
                isinstance(visuals, (list, tuple)) and (isinstance(visuals[0], (np.ndarray)))
            )
            if not visuals_is_batched:
                visuals = [visuals]

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_visual = []
        all_acoustic = []
        all_label_ids = []

        for (index, words) in enumerate(word_list):
            acoustic = acoustics[index]
            visual = visuals[index]
            label_id = label_ids[index]

            tokens, inversions = [], []
            for idx, word in enumerate(words):
                tokenized = self.tokenizer.tokenize(word)
                tokens.extend(tokenized)
                inversions.extend([idx] * len(tokenized))

            # Check inversion
            assert len(tokens) == len(inversions)

            aligned_visual = []
            aligned_audio = []

            for inv_idx in inversions:
                aligned_visual.append(visual[inv_idx, :])
                aligned_audio.append(acoustic[inv_idx, :])

            visual = np.array(aligned_visual)
            acoustic = np.array(aligned_audio)

            # Truncate input if necessary
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[: max_seq_length - 2]
                acoustic = acoustic[: max_seq_length - 2]
                visual = visual[: max_seq_length - 2]

            input_ids, visual, acoustic, input_mask, segment_ids = self.prepare_bert_input(
                tokens, visual, acoustic, max_seq_length, visual_dim, acoustic_dim
            )

            # Check input length
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert acoustic.shape[0] == max_seq_length
            assert visual.shape[0] == max_seq_length

            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
            all_visual.append(visual)
            all_acoustic.append(acoustic)
            all_label_ids.append(label_id)

        inputs = BatchEncoding({'input_ids': all_input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, 'visual': visual, 'acoustic': acoustic, 'label_id': label_id})
        inputs = inputs.convert_to_tensors(return_tensors)
        return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def prepare_bert_input(self, tokens, visual, acoustic, max_seq_length, visual_dim, acoustic_dim):
        CLS = self.tokenizer.cls_token
        SEP = self.tokenizer.sep_token
        tokens = [CLS] + tokens + [SEP]

        # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
        acoustic_zero = np.zeros((1, acoustic_dim))
        acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
        visual_zero = np.zeros((1, visual_dim))
        visual = np.concatenate((visual_zero, visual, visual_zero))

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        pad_length = max_seq_length - len(input_ids)

        acoustic_padding = np.zeros((pad_length, acoustic_dim))
        acoustic = np.concatenate((acoustic, acoustic_padding))

        visual_padding = np.zeros((pad_length, visual_dim))
        visual = np.concatenate((visual, visual_padding))

        padding = [0] * pad_length

        # Pad inputs
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        return input_ids, visual, acoustic, input_mask, segment_ids