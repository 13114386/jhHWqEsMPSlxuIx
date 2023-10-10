from __future__ import unicode_literals, print_function, division
import os, json
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput
from file_utils import ChunkSaver
from buildtool.annotated_feature_flatter import AnnotatedFeatureFlatter
from buildtool.annotated_word_segment_masker import (AnnotatedWordSegmentMasker,
                                                     DocIterator)
from buildtool.annotated_utils import (AnnotateSentenceIteratorByField)
from buildtool.annotated_doc_wordlist_encoder import AnnotatedDocEncoder
from buildtool.subword_graph_annotator import SubwordGraphAnnotator


class AnnotatedDocExtractor():
    def __init__(
        self,
        first_sentence_only, 
        lower_case,
        count_all_tokens=True,
        aws_masker=None
    ):
        self.first_sentence_only = first_sentence_only
        self.lower_case = lower_case
        self.count_all_tokens = count_all_tokens
        self.aws_masker = aws_masker

    def __call__(self, line, doc, logger):
        annotation = json.loads(line, strict=False)
        doc_feat = self.get_token_document(annotation, doc, self.first_sentence_only,
                                           self.lower_case, logger)
        return ModelOutput(doc_feat)

    def get_token_document(self, annotation, doc, first_sentence_only, lower_case, logger):
        titer = AnnotateSentenceIteratorByField(annotation, "tokens",
                                                first_sentence_only=first_sentence_only)
        document = []
        sent_sizes = []
        for tokens in titer: # Iterate tokens of each sentence
            if self.count_all_tokens:
                all_count = sum([len(t["originalText"].strip().split(" ")) for t in tokens])
                sent_sizes.append(all_count)
            else:
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


class AnnotatedDocFlatter():
    def __init__(self):
        super().__init__()

    def __call__(self, doc, doc_mask, doc_sentence_sizes, tk_doc, logger):
        doc = AnnotatedFeatureFlatter.flatten_doc(doc, logger)
        doc_size = len(doc.split(" "))

        if sum(doc_sentence_sizes) != doc_size:
            raise ValueError(f"Annotated sentences mismatch doc: {sum(doc_sentence_sizes)} != {doc_size}")

        doc_mask = AnnotatedFeatureFlatter.flatten_feature(doc_mask, logger)
        if len(doc_mask) != doc_size:
            raise ValueError(f"Annotated token mismatch doc: {len(doc_mask)} != {doc_size}")

        input_ids, attention_masks, token_masks = None, None, None
        if tk_doc:
            input_ids, attention_masks, token_masks = AnnotatedFeatureFlatter.flatten_tk_doc(tk_doc, logger)
            if sum(token_masks) != doc_size + 2: # Count BOS and EOS
                raise ValueError(f"Tokens mismatch doc: {sum(token_masks)} != {doc_size + 2}")

        return {"doc":doc,
                "doc_mask": doc_mask,
                "input_ids":input_ids,
                "attention_mask":attention_masks,
                "token_mask":token_masks}


class AnnotatedDocComposer():
    def __init__(self, config, tokenizer, aws_masker):
        super().__init__()
        self.doc_extractor = AnnotatedDocExtractor(
                                first_sentence_only=config.first_sentence_only,
                                lower_case=config.lower_case,
                                count_all_tokens=config.count_all_tokens,
                                aws_masker=aws_masker
                            )
        self.doc_encoder = AnnotatedDocEncoder(
                                tokenizer=tokenizer,
                                leading_space_word=config.leading_space_word,
                                pad_to_max_length=config.pad_to_max_length,
                                ignore_pad_token_for_loss=config.ignore_pad_token_for_loss,
                                max_length=config.max_len,
                                truncation=config.truncation,
                            )
        self.doc_flatter = AnnotatedDocFlatter()
        self.subword_grapher = SubwordGraphAnnotator(
                                  depth_padding_id=config.subword_depth_padding_id,
                                  mask_all=config.subword_mask_all,
                                  self_loop=config.subword_self_loop)

    def __call__(self, line, doc, logger):
        features = self.doc_extractor(line, doc, logger=logger)
        data = self.doc_flatter(doc=features.tdoc,
                                doc_mask=features.tdoc_mask,
                                doc_sentence_sizes=features.sent_sizes,
                                tk_doc=None,
                                logger=logger)
        encoded_doc = self.doc_encoder([data["doc"]], [data["doc_mask"]],
                                       sent_sizes=[features.sent_sizes],
                                       logger=logger)
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


class AnnotatedPairing():
    def __init__(self):
        super().__init__()

    def __call__(self, config, logger):
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name,
                                                  use_fast=not config.use_slow_tokenizer)
        for split_type in config.split_types:
            atcl_files = self.get_source_files(split_type=split_type,
                                            pair_type=config.pair_types["article"],
                                            src_dir=config.src_dir,
                                            output_dir=config.output_dir,
                                            source_stem=config.source_stem,
                                            output_stem=config.output_stem)
            hlit_files = self.get_source_files(split_type=split_type,
                                            pair_type=config.pair_types["summary"],
                                            src_dir=config.src_dir,
                                            output_dir=config.output_dir,
                                            source_stem=config.source_stem,
                                            output_stem=config.output_stem)
            os.makedirs(config.output_dir, exist_ok=True)
            self.process(config=config, split_type=split_type,
                         atcl_files=atcl_files, hlit_files=hlit_files,
                         tokenizer=tokenizer, logger=logger)

    def process(
        self,
        config,
        split_type,
        atcl_files,
        hlit_files,
        tokenizer,
        logger
    ):
        downloaded_path = os.path.join(config.downloaded_dir,
                                    config.downloaded_file.format(split_type=split_type))
        skip_index_path = os.path.join(config.src_dir,
                                    config.skip_index_file.format(split_type=split_type))
        doc_iter = DocIterator(
                        article_fieldname=config.pair_types["article"],
                        summary_fieldname=config.pair_types["summary"],
                        skip_line_path=skip_index_path
                    )
        atcl_aws_masker = AnnotatedWordSegmentMasker()
        atcl_afc = AnnotatedDocComposer(config, tokenizer, atcl_aws_masker)
        hlit_aws_masker = AnnotatedWordSegmentMasker()
        hlit_afc = AnnotatedDocComposer(config, tokenizer, hlit_aws_masker)

        overflow_lines = []
        error_lines = []
        composed = {"atcl": [], "hlit": []}
        save_ids = {"atcl": atcl_files.output,
                    "hlit": hlit_files.output}
        with ChunkSaver(save_ids, config.chunk_size, convert_json=False) as saver, \
            open(atcl_files.annotation, "r", encoding="utf-8") as atcl_fp, \
            open(hlit_files.annotation, "r", encoding="utf-8") as hlit_fp, \
            open(downloaded_path, "r", encoding="utf-8") as doc_fp:
            for index, (atcl_line, hlit_line) in enumerate(zip(atcl_fp, hlit_fp)):
                try:
                    (atcl_doc, hlit_doc) = next(doc_iter(doc_fp))
                    atcl_data = atcl_afc(atcl_line, atcl_doc, logger)
                    hlit_data = hlit_afc(hlit_line, hlit_doc, logger)
                except Exception as ex:
                    error_lines.append(index)
                    logger.error(str(ex))
                    continue

                ok = config.max_len == -1 or \
                    (len(atcl_data["input_ids"]) <= config.max_len and \
                     len(hlit_data["input_ids"]) <= config.max_len)
                if not ok:
                    overflow_lines.append(index)
                    continue

                if not config.should_output_doc:
                    if "doc" in atcl_data:
                        del atcl_data["doc"]
                    if "doc" in hlit_data:
                        del hlit_data["doc"]

                try:
                    atcl_data_encoded = json.dumps(atcl_data, ensure_ascii=False)
                    hlit_data_encoded = json.dumps(hlit_data, ensure_ascii=False)
                    composed["atcl"].append(atcl_data_encoded)
                    composed["hlit"].append(hlit_data_encoded)
                except (json.decoder.JSONDecodeError, UnicodeEncodeError) as ex:
                    error_lines.append(index)
                    logger.error(str(ex))
                    continue

                if saver(features=composed, index=index, last_save=False):
                    [composed[k].clear() for k, v in composed.items()]
            # If any remaining
            if saver(features=composed, index=index, last_save=True):
                [composed[k].clear() for k, v in composed.items()]

        if len(overflow_lines):
            logger.warning(f"{split_type} has {len(overflow_lines)} very long articles of indexed list {overflow_lines}")
        if len(error_lines):
            logger.warning(f"{split_type} has {len(error_lines)} improperly parsed articles of indexed list {error_lines}")

    def get_source_files(self, split_type, pair_type, src_dir, output_dir,
                         source_stem, output_stem):
        source_file = f"{split_type}.{pair_type}{source_stem}"
        source_filepath = os.path.join(src_dir, source_file)
        output_filepath = os.path.join(output_dir, f"{split_type}.{pair_type}{output_stem}")
        return ModelOutput({"annotation": source_filepath,
                            "output": output_filepath})


from parse_args import AnnotatedDocArgsParse
def main():
    import logging
    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger(__name__)

    args = AnnotatedDocArgsParse()()

    split_types = args.split_types
    if split_types is not None:
        split_types = split_types.strip(" []")
        split_types = split_types.split(",")

    pair_types = args.pair_types
    if pair_types is not None:
        pair_types = pair_types.strip(" []")
        pair_types = pair_types.split(",")
        pair_types = {"article": pair_types[0], "summary": pair_types[1]}

    # Build dataset
    cfg = ModelOutput(json.loads(args.compose))
    cfg.count_all_tokens = args.count_all_tokens
    if args.downloaded_dir is not None:
        cfg.downloaded_dir = args.downloaded_dir
    if args.src_dir is not None:
        cfg.src_dir = args.src_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if split_types is not None:
        cfg.split_types = split_types
    if pair_types is not None:
        cfg.pair_types = pair_types
    if args.tokenizer_name is not None:
        cfg.tokenizer_name = args.tokenizer_name
    pairing = AnnotatedPairing()
    pairing(cfg, logger)
    logger.info("Done!")

if __name__ == "__main__":
    main()
