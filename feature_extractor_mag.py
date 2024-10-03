"""
Feature extractor class for MAG-BERT-ARL (eGeMAPS)
"""

from typing import List, Optional, Union

import numpy as np
import opensmile

from transformers.utils.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.utils.feature_extraction_utils import BatchFeature

class MAGBertFeatureExtractor(SequenceFeatureExtractor):
    """
    Constructs a MAG-BERT feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
            """

    model_input_names = ["input_ids", "acoustic", "visual", "token_type_ids", "attention_mask", "labels"]

    def __init__(
        self,
        feature_size=88,
        sampling_rate=16000,
        padding_value=0.0,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    @staticmethod
    def replace_nans_with_avg(row: List[any]) -> List[np.ndarray]:
        avg_row = np.nanmean(row, axis=0)  # Compute the mean while ignoring NaNs
        nan_mask = np.isnan(row)  # Mask for NaN values
        row[nan_mask] = np.take(avg_row, np.where(nan_mask)[1])  # Replace NaNs with the corresponding column averages
        return row

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[np.ndarray]],
        segments: Union[List[List[float]], List[List[int]], List[List[str]], List[List[List[float]]], List[List[List[int]]], List[List[List[str]]]],
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                Wav2Vec2 models that have set `config.feat_extract_norm == "group"`, such as
                [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), have **not** been trained using
                `attention_mask`. For such models, `input_values` should simply be padded with 0 and no
                `attention_mask` should be passed.

                For Wav2Vec2 models that have set `config.feat_extract_norm == "layer"`, such as
                [wav2vec2-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self), `attention_mask` should
                be passed for batched inference.

                </Tip>

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            padding_value (`float`, *optional*, defaults to 0.0):
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray)))
        )
        if is_batched and isinstance(segments[0][0], (float, int, str)):
            raise ValueError(f"The type for the segments is not batched, should be either List[List[List[float]]], List[List[List[int]]], or List[List[List[str]]]")

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]
            segments = [segments]

        assert len(raw_speech) == len(segments)

        # Extract features using opensmile
        feature_list = []
        for signal, segment in zip(raw_speech, segments):
            segment_features_list = []
            for start, end in segment:
                features = smile.process_signal(signal, sampling_rate, start=start, end=end)
                features = BatchFeature({'features': features.values.tolist()}.convert_to_tensors(return_tensors))
                segment_features_list.append(features['features'])
            segment_features_list = replace_nans_with_avg(segment_features_list)
            segment_features_list = BatchFeature({'segment_features_list': segment_features_list}).convert_to_tensors(return_tensors)
            feature_list.append(segment_features_list['segment_features_list'])

        # convert into correct format
        encoded_inputs = BatchFeature({"acoustic": feature_list}).convert_to_tensors(return_tensors)

        return encoded_inputs