from __future__ import unicode_literals, print_function, division
import os, abc
from itertools import chain
from collections import Counter
import torch
from file_utils import saveToJson, loadFromJson

class Vocabase(abc.ABC):
    def __init__(self, special_tokens=None):
        self._vocab = Counter()
        self.special_tokens = special_tokens

    @property
    def specials(self):
        return self.special_tokens

    @property
    def vocab(self):
        return self._vocab

    @vocab.setter
    def vocab(self, value):
        try:
            self._vocab.update(value)
        except TypeError:
            self._vocab.update(chain.from_iterable(value))

    @staticmethod
    def save(split_type, file_dir, save_key, vocab, specials, logger, ext=".json"):
        i2w = list(specials)+list(vocab.keys()) if specials else list(vocab.keys())
        result_vocab = {
            "specials": specials,
            "freqs": vocab,
            "i2w": i2w,
            "w2i": {w: i for i, w in enumerate(i2w)}
        }
        vocab_name = f"{split_type}.{save_key}{ext}" if split_type is not None \
                    else f"{save_key}{ext}"
        if isinstance(file_dir, str):
            file_dir = [file_dir]
        filepaths = [os.path.join(f, vocab_name) for f in file_dir]
        return_maps = []
        for filepath in filepaths:
            logger.info("Save {} vocab to {}".format(save_key, filepath))
            # Save json version for auditing.
            saveToJson(filepath, result_vocab)
            # Save a pickled version for runtime.
            fpart, _ = os.path.splitext(filepath)
            torch.save(result_vocab, f'{fpart}.pth')
            return_maps.append({"path": filepath, "data": result_vocab})
        return return_maps

    @staticmethod
    def load(file_dir, vocab_info, to_pt_vocab=True):
        vocab_map = {}
        for key, val in vocab_info.items():
            vocab_path = os.path.join(file_dir, val)
            vocab = Vocabase.load_from_file(vocab_path)
            vocab = Vocabase.pt_vocab(vocab) if to_pt_vocab else vocab
            vocab_map[key] = vocab
        return vocab_map

    @staticmethod
    def load_from_file(filepath):
        # Load pickled version if exist
        vocab = None
        fpart, _ = os.path.splitext(filepath)
        if os.path.isfile(f'{fpart}.pth'):
            vocab = torch.load(f'{fpart}.pth')
        # Fall through for json version
        if vocab is None:
            vocab = loadFromJson(filepath)
        return vocab

    @staticmethod
    def pt_vocab(vocab_json):
        from collections import Counter
        from torchtext.vocab import vocab
        return vocab(Counter(vocab_json["freqs"]), specials=vocab_json["specials"])


class AnnotatedFeatureVocabBuilder(Vocabase):
    def __init__(self, special_tokens=None):
        super().__init__(special_tokens)


class AnnotatedNumberFeatureStat():
    def __init__(self, plus_one=True):
        self._size = 0
        self._plus_one = int(plus_one)
        self._freq = Counter()

    @property
    def size(self):
        return self._size + self._plus_one

    @property
    def stat(self):
        freq = dict(self._freq.most_common())
        return freq

    @stat.setter
    def stat(self, value):
        if isinstance(value, list) and isinstance(value[0], list):
            flattened = list(chain.from_iterable(value))
        else:
            flattened = value
        self._freq.update(flattened)
        value = max(flattened)
        if self._size < value:
            self._size = value

    @staticmethod
    def save(file_dir, name, save_map, logger=None, ext=".json"):
        filepath = os.path.join(file_dir, f"{name}{ext}")
        saveToJson(filepath, save_map)
