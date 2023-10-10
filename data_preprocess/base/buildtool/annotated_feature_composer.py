from __future__ import unicode_literals, print_function, division
from typing import Dict, List
from buildtool.annotated_feature_extractor import AnnotatedFeatureExtractor
from buildtool.annotated_doc_wordlist_encoder import AnnotatedDocEncoder
from buildtool.annotated_feature_flatter import AnnotatedFeatureFlatter
from buildtool.subword_graph_annotator import SubwordGraphAnnotator

sanity_check=True

def check_sentence_stats(sent_stats: Dict, sent_sizes: List):
    import functools
    sent_stats = [b[-1] for b in sorted(sent_stats.items())]
    return functools.reduce(lambda x, y : x and y, map(lambda p, q: p == q, sent_stats, sent_sizes), True)

class AnnotatedFeatureComposer():
    def __init__(
        self,
        config,
        tokenizer,
        aws_masker,
        use_corefs=False,
        use_ner=False
    ):
        super().__init__()
        self.feature_extractor = AnnotatedFeatureExtractor(
                                    first_sentence_only=config.first_sentence_only,
                                    missing_token=config.missing_token,
                                    lower_case=config.lower_case,
                                    self_loop_root=config.self_loop_root,
                                    use_corefs=use_corefs,
                                    use_ner=use_ner,
                                    aws_masker=aws_masker
                                )
        self.doc_encoder = AnnotatedDocEncoder(
                                tokenizer=tokenizer,
                                leading_space_word=config.leading_space_word,
                                pad_to_max_length=config.pad_to_max_length,
                                ignore_pad_token_for_loss=config.ignore_pad_token_for_loss
                            )
        self.feature_flatter = AnnotatedFeatureFlatter()
        self.subword_grapher = SubwordGraphAnnotator(
                                    depth_padding_id=config.subword_depth_padding_id,
                                    mask_all=config.subword_mask_all,
                                    self_loop=config.subword_self_loop
                                )

    def __call__(self, line, doc, logger):
        features = self.feature_extractor(line, doc, logger=logger)

        if sanity_check \
            and features.get("sentencestat", None) is not None:
            assert check_sentence_stats(features["sentencestat"], features.sent_sizes), \
                    "Feature composer found sentence parsing inconsistence."

        data = self.feature_flatter(
                    doc=features.tdoc,
                    doc_mask=features.tdoc_mask,
                    doc_sentence_sizes=features.sent_sizes,
                    tk_doc=None,
                    pos=features.pos,
                    head=features.head,
                    root_head_mask=features.root_head_mask,
                    inarc=features.inarc,
                    depth=features.depth_d,
                    corefs=features.get("corefs", None),
                    logger=logger
                )

        encoded_doc = self.doc_encoder(
                        [data["doc"]],
                        [data["doc_mask"]],
                        sent_sizes=[features.sent_sizes],
                        logger=logger
                    )
        encoded_doc = encoded_doc[0]
        del data["doc_mask"] # Job is done. So, delete it.
        data["input_ids"] = encoded_doc["input_ids"]
        data["attention_mask"] = encoded_doc["attention_mask"]
        data["token_mask"] = encoded_doc["token_mask"]
        data["tokenized_sent_sizes"] = encoded_doc["tokenized_sent_sizes"]
        subwords = self.subword_grapher(data["token_mask"])
        data["subword_edge"] = subwords["edge"]
        data["subword_depth"] = subwords["depth"]
        data["subword_mask"] = subwords["mask"]
        data["subword_span"] = subwords["span"]
        data["subword_span_map"] = subwords["span_map"]
        return data
