from __future__ import unicode_literals, print_function, division
from abc import ABC, abstractmethod
import argparse


class ArgsParseBase(ABC):
    def __call__(self):
        parser = self.pre_parse()
        args = self.parse_args(parser)
        args = self.post_parse(args)
        return args

    @abstractmethod
    def pre_parse(self):
        raise NotImplementedError("ArgsParseBase.parse_args not implemented.")

    def parse_args(self, parser):
        args = parser.parse_args()
        return args

    def post_parse(self, args):
        return args


class TrainDataArgsParse(ArgsParseBase):
    def pre_parse(self):
        parser = argparse.ArgumentParser(description="Build train and validation dataset")

        parser.add_argument('--downloaded_dir', type=str, default=None)
        parser.add_argument('--src_dir', type=str, default=None)
        parser.add_argument('--output_dir', type=str, default=None)
        parser.add_argument('--split_types', type=str, default=None)
        parser.add_argument('--pair_types', type=str, default=None)
        parser.add_argument('--tokenizer_name', type=str, default=None)
        parser.add_argument('--build_vocab', action='store_true')
        parser.add_argument('--build_compose', action='store_true')
        parser.add_argument('--reconcile_vocab', action='store_true')
        parser.add_argument('--build_stype2id', action='store_true')
        parser.add_argument('--use_source_corefs', action='store_true')
        parser.add_argument('--use_target_corefs', action='store_true')
        parser.add_argument('--use_entity_links', action='store_true')
        parser.add_argument('--analysis_stem', type=str, default=None)

        parser.add_argument('--compose', type=str, required=False,
                            default='{"src_dir": "D:/Dev/Projects/Data/nlp/cnndm/corenlp.parse.ner_coref", \
                                        "output_dir": "D:/Dev/Projects/Data/nlp/cnndm/struct.ner_coref", \
                                        "downloaded_dir": "D:/Dev/Projects/Data/nlp/cnndm/downloaded", \
                                        "downloaded_file": "{split_type}.json", \
                                        "skip_index_file": "error.{split_type}.log", \
                                        "source_stem": ".json", \
                                        "split_types": ["train", "validation"], \
                                        "pair_types": {"article": "article", "summary": "highlights"}, \
                                        "first_sentence_only": false, \
                                        "missing_token": "<unk>", \
                                        "lower_case": false, \
                                        "self_loop_root": true, \
                                        "pad_to_max_length": false, \
                                        "ignore_pad_token_for_loss": true, \
                                        "max_len": 1024, \
                                        "chunk_size": 100, \
                                        "tokenizer_name": "facebook/bart-base", \
                                        "leading_space_word": true, \
                                        "use_slow_tokenizer": false, \
                                        "subword_depth_padding_id": 0, \
                                        "subword_mask_all": false, \
                                        "subword_self_loop": false, \
                                        "should_output_doc": false, \
                                        "vocabs": { "pos.vocab": "pos.vocab.json", \
                                                    "inarc.vocab": "inarc.vocab.json", \
                                                    "coref.type.vocab": "coref.type.vocab.json", \
                                                    "coref.number.vocab": "coref.number.vocab.json", \
                                                    "coref.animacy.vocab": "coref.animacy.vocab.json", \
                                                    "coref.gender.vocab": "coref.gender.vocab.json" }, \
                                        "vocab_specials": ["<pad>", "<unk>"], \
                                        "output_stem": ".compose.json", \
                                        "analysis_stem": null, \
                                        "coref_entity_max_len": -1 }',
                            help='Compose dataset.')

        parser.add_argument('--stype2id', type=str, required=False,
                            default='{"src_dir": "D:/Dev/Projects/Data/nlp/cnndm/struct.ner_coref", \
                                    "output_dir": "D:/Dev/Projects/Data/nlp/cnndm/struct.ner_coref", \
                                    "split_types": ["train", "validation"], \
                                    "pair_types": {"article": "article", "summary": "highlights"}, \
                                    "vocabs": { "pos.vocab": "pos.vocab.json", \
                                                "inarc.vocab": "inarc.vocab.json" }, \
                                    "coref_vocabs": { "coref.type.vocab": "coref.type.vocab.json", \
                                                "coref.number.vocab": "coref.number.vocab.json", \
                                                "coref.animacy.vocab": "coref.animacy.vocab.json", \
                                                "coref.gender.vocab": "coref.gender.vocab.json" }, \
                                    "unk_token": "<unk>", \
                                    "chunk_size": 100, \
                                    "input_stem": ".compose.json", \
                                    "output_stem": ".dataset.json" }',
                            help='Replace feature value by vocab id.')
        return parser


class AnnotatedDocArgsParse(ArgsParseBase):
    def pre_parse(self):
        parser = argparse.ArgumentParser(description="Build train and validation dataset - doc only")
        parser.add_argument('--downloaded_dir', type=str, default=None)
        parser.add_argument('--src_dir', type=str, default=None)
        parser.add_argument('--output_dir', type=str, default=None)
        parser.add_argument('--split_types', type=str, default=None)
        parser.add_argument('--pair_types', type=str, default=None)
        parser.add_argument('--tokenizer_name', type=str, default=None)
        parser.add_argument('--count_all_tokens', type=bool, default=True)

        parser.add_argument(
            '--compose', type=str, required=False,
                        default='{"src_dir": "./", \
                                "output_dir": "./", \
                                "downloaded_dir": "./downloaded", \
                                "downloaded_file": "{split_type}.json", \
                                "skip_index_file": "error.{split_type}.log", \
                                "source_stem": ".json", \
                                "split_types": ["test"], \
                                "pair_types": {"article": "article", "summary": "highlights"}, \
                                "first_sentence_only": false, \
                                "lower_case": false, \
                                "pad_to_max_length": false, \
                                "ignore_pad_token_for_loss": true, \
                                "max_len": 1024, \
                                "truncation": "longest_first", \
                                "chunk_size": 100, \
                                "tokenizer_name": "facebook/bart-base", \
                                "leading_space_word": true, \
                                "use_slow_tokenizer": false, \
                                "subword_depth_padding_id": 0, \
                                "subword_mask_all": false, \
                                "subword_self_loop": false, \
                                "should_output_doc": false, \
                                "count_all_tokens": true, \
                                "output_stem": ".doc.dataset.json"}',
                        help='Compose doc dataset.'
        )
        return parser
