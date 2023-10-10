from __future__ import unicode_literals, print_function, division
import json
from transformers.file_utils import ModelOutput
from buildtool.corenlp_preprocess_handlers import *
from buildtool.annotated_utils import (
    AnnotateSentenceIteratorByField,
    AnnotateCorefsIterator,
    AnnotateSentenceIteratorByFields
)


class AnnotatedFeatureExtractor():
    def __init__(
        self,
        first_sentence_only, missing_token, lower_case,
        self_loop_root=True,
        use_corefs=False,
        use_ner=False,
        aws_masker=None
    ):
        self.constituency = ConstituencyParsingProcess()
        self.dependency = DependencyParsingProcess()
        self.coreference = CorefParsingProcess() if use_corefs else None
        self.ner = NerParsingProcess() if use_ner else None
        self.first_sentence_only = first_sentence_only
        self.missing_token = missing_token
        self.lower_case = lower_case
        self.self_loop_root = self_loop_root
        self.aws_masker = aws_masker

    def __call__(self, line, doc, logger):
        annotation = json.loads(line, strict=False)
        doc_feat = self.get_token_document(
                        annotation,
                        doc,
                        self.first_sentence_only,
                        self.lower_case,
                        logger
                    )
        con_feat = self.get_constituency_feature(
                        annotation,
                        self.first_sentence_only
                    )
        dep_feat = self.get_dependency_feature(
                        annotation,
                        self.first_sentence_only,
                        self.missing_token,
                        self.self_loop_root
                    )
        coref_feat = {}
        if self.coreference:
            coref_feat = self.get_coref_feature(
                            annotation,
                            first_sentence_only=self.first_sentence_only
                        )
        ner_feat = {}
        if self.ner:
            ner_feat = self.get_ner_feature(
                            annotation,
                            first_sentence_only=self.first_sentence_only
                        )
        return ModelOutput({**doc_feat,
                            **con_feat,
                            **dep_feat,
                            **coref_feat,
                            **ner_feat})

    def get_token_document(self, annotation, doc, first_sentence_only, lower_case, logger):
        titer = AnnotateSentenceIteratorByField(annotation, "tokens",
                                                first_sentence_only=first_sentence_only)
        document = []
        sent_sizes = []
        for tokens in titer: # Iterate tokens of each sentence
            sent_sizes.append(len(tokens))
            # Concatenate tokens of a sentence
            sentence = " ".join([token["originalText"].lower() \
                                    if lower_case else token["originalText"] \
                                        for token in tokens])
            document.append([sentence.strip()])
        mask = None
        if self.aws_masker:
            mask = self.aws_masker(document, doc, logger)
        return {"tdoc":document, "tdoc_mask":mask, "sent_sizes":sent_sizes}

    def get_constituency_feature(self, annotation, first_sentence_only):
        # Pos Tag
        titer = AnnotateSentenceIteratorByField(annotation, "parse",
                                                first_sentence_only=first_sentence_only)
        pos = []
        depth_c = []
        for data in titer:
            pos_i, depth_c_i = self.constituency(data)
            pos.append(pos_i)
            depth_c.append(depth_c_i)
        return {"pos":pos, "depth_c":depth_c}

    def get_dependency_feature(self, annotation, first_sentence_only,
                               missing_token, self_loop_root):
        def set_self_loop(parent):
            neg1_index = parent.index(-1)
            parent[neg1_index] = neg1_index
            return parent

        # Dependency
        titer = AnnotateSentenceIteratorByField(annotation, "basicDependencies",
                                                first_sentence_only=first_sentence_only)
        inarc = []
        out_arc = []
        depth_d = []
        parent = []
        root_parent_mask = [] # Mask root position
        for data in titer:
            inarc_i, out_arc_i, depth_d_i, parent_i, root_i = self.dependency(data, missing_token)
            inarc.append(inarc_i)
            out_arc.append(out_arc_i)
            depth_d.append(depth_d_i)
            root_pos = root_i
            assert root_pos == parent_i.index(-1)
            mask = [1]*len(parent_i)
            mask[root_pos] = 0
            root_parent_mask.append(mask)
            if self_loop_root:
                parent_i = set_self_loop(parent_i)
            parent.append(parent_i)
        return {"inarc":inarc, "depth_d":depth_d,
                "head":parent, "root_head_mask":root_parent_mask,
                "out_arc":out_arc}

    def get_coref_feature(self, annotation, first_sentence_only):
        titer = AnnotateCorefsIterator(annotation=annotation,
                                        first_sentence_only=first_sentence_only)
        corefs = []
        for data in titer:
            coref = self.coreference(data)
            corefs.append(coref)
        if len(corefs) == 0:
            raise ValueError("No coreference is available.")
        return {"corefs": corefs}

    def get_ner_feature(self, annotation, first_sentence_only):
        sentence_fields = ["index", "entitymentions", "tokens"]
        iter = AnnotateSentenceIteratorByFields(
                    annotation,
                    sentence_fields,
                    first_sentence_only=first_sentence_only
                )
        all_mentioned = []
        sentence_stats = {}
        for sentence in iter:
            ner = self.ner(sentence)
            sentence_stats[ner["sentenceindex"]] = ner["sentencelength"]
            if len(ner["entitymentions"]) > 0:
                all_mentioned.append({
                    "sentenceindex": ner["sentenceindex"],
                    "entitymentions": ner["entitymentions"],
                })
        return {"ners": all_mentioned,
                "sentencestat": sentence_stats}
