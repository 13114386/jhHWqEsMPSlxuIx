from __future__ import unicode_literals, print_function, division
import itertools
from operator import add
from buildtool.annotated_utils import cumulative
from buildtool.annotated_coref_data_utils import (merge_coref_by_antecedent,
                                                merge_all_corefs)

class AnnotatedFeatureFlatter():
    def __init__(self):
        super().__init__()

    def __call__(self, doc, doc_mask, doc_sentence_sizes, tk_doc,
                 pos, head, root_head_mask, inarc, depth,
                 corefs, logger):
        doc = self.flatten_doc(doc, logger)
        doc_size = len(doc.split(" "))

        if sum(doc_sentence_sizes) != doc_size:
            raise ValueError(f"Annotated sentences mismatch doc: {sum(doc_sentence_sizes)} != {doc_size}")

        doc_mask = self.flatten_feature(doc_mask, logger)
        if len(doc_mask) != doc_size:
            raise ValueError(f"Annotated token mismatch doc: {len(doc_mask)} != {doc_size}")

        input_ids, attention_masks, token_masks = None, None, None
        if tk_doc:
            input_ids, attention_masks, token_masks = self.flatten_tk_doc(tk_doc, logger)
            if sum(token_masks) != doc_size + 2: # Count BOS and EOS
                raise ValueError(f"Tokens mismatch doc: {sum(token_masks)} != {doc_size + 2}")

        pos = self.flatten_feature(pos, logger)
        if len(pos) != doc_size:
            raise ValueError(f"POS mismatch doc: {len(pos)} != {doc_size}")

        head = self.flatten_feature_by_offset(head, logger)
        if len(head) != doc_size:
            raise ValueError(f"Head mismatch doc: {len(head)} != {doc_size}")

        root_head_mask = self.flatten_feature(root_head_mask, logger)
        if len(root_head_mask) != doc_size:
            raise ValueError(f"Root head mask mismatch doc: {len(root_head_mask)} != {doc_size}")

        inarc = self.flatten_feature(inarc, logger)
        if len(inarc) != doc_size:
            raise ValueError(f"inarc mismatch doc: {len(inarc)} != {doc_size}")

        depth = self.flatten_feature(depth, logger)
        if len(depth) != doc_size:
            raise ValueError(f"Depth mismatch doc: {len(depth)} != {doc_size}")

        corefs_data = {}
        if corefs:
            corefs_data = self.flatten_coref_feature(corefs, doc_sentence_sizes, logger)

        return {**{"doc":doc,
                    "doc_mask": doc_mask,
                    "sentence_sizes": doc_sentence_sizes,
                    "input_ids":input_ids,
                    "attention_mask":attention_masks,
                    "token_mask":token_masks,
                    "pos":pos, "head":head,
                    "root_head_mask":root_head_mask,
                    "inarc":inarc,
                    "depth_d":depth},
                **corefs_data}

    @staticmethod
    def flatten_feature_by_offset(feature, logger):
        if isinstance(feature, str):
            feature = eval(feature)
        values = [len(feat) for feat in feature]
        cumsum = cumulative(values)
        cumsum = cumsum[:len(feature)]
        # Offset
        for i, (aa, cc) in enumerate(zip(feature, cumsum)):
            feature[i] = list(map(add, aa, [cc]*len(aa)))
        # Flatten
        feature = list(itertools.chain(*feature))
        return feature

    @staticmethod
    def flatten_feature(feature, logger):
        if isinstance(feature, str):
            feature = eval(feature)
        feature = list(itertools.chain(*feature))
        return feature

    @staticmethod
    def flatten_tk_doc(doc, logger):
        if isinstance(doc, str):
            doc = eval(doc)
        size = len(doc)
        input_ids = []
        attention_masks = []
        token_masks = []
        for i, sentence in enumerate(doc):
            if i == 0:
                input_ids += sentence[0]["input_ids"][:-1]
                attention_masks += sentence[0]["attention_mask"][:-1]
                token_masks += sentence[0]["token_mask"][:-1]
            elif i == size - 1:
                input_ids += sentence[0]["input_ids"][1:]
                attention_masks += sentence[0]["attention_mask"][1:]
                token_masks += sentence[0]["token_mask"][1:]
            else:
                input_ids += sentence[0]["input_ids"][1:-1]
                attention_masks += sentence[0]["attention_mask"][1:-1]
                token_masks += sentence[0]["token_mask"][1:-1]
        return input_ids, attention_masks, token_masks

    @staticmethod
    def flatten_doc(doc, logger):
        if isinstance(doc, str):
            doc = eval(doc)
        doc = list(itertools.chain(*doc))
        doc = " ".join(doc)
        return doc

    @staticmethod
    def flatten_coref_feature(corefs, sentence_sizes, logger):
        coref_by_antecedent = merge_coref_by_antecedent(corefs, sentence_sizes, logger=logger)
        merged_all = merge_all_corefs(coref_by_antecedent)
        return merged_all
