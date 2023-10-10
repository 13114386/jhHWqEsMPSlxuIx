from __future__ import unicode_literals, print_function, division
import os, json, glob, copy
from collections import Counter
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput
from file_utils import ChunkSaver
from buildtool.annotated_feature_composer import AnnotatedFeatureComposer
from buildtool.annotated_feature_vocab import (
    AnnotatedFeatureVocabBuilder,
    AnnotatedNumberFeatureStat,
    Vocabase
)
from buildtool.annotated_word_segment_masker import (
    AnnotatedWordSegmentMasker,
    DocIterator
)


class VocabReconciler():
    def __call__(self, config, logger):
        for k, v in config.vocabs.items():
            all_vs = f"*.{v}"
            self.reconcile(
                all_vs,
                key=k,
                src_dir=config.src_dir,
                output_dir=config.output_dir,
                logger=logger
            )

    def reconcile(self, vs, key, src_dir, output_dir, logger):
        _, ext = os.path.splitext(vs)
        subdirs = [x[0] for x in os.walk(src_dir)]
        filepaths = []
        for subdir in subdirs:
            src_pat = os.path.join(subdir, vs)
            fpaths = glob.glob(src_pat)
            filepaths += fpaths
        m_vocab = {}
        for filepath in filepaths:
            vocab = Vocabase.load_from_file(filepath)
            if "specials" not in m_vocab:
                m_vocab["specials"] = copy.deepcopy(vocab["specials"])
            m_vocab["freqs"] = dict(Counter(m_vocab["freqs"])+Counter(vocab["freqs"])) \
                                if "freqs" in m_vocab else copy.deepcopy(vocab["freqs"])
        if len(m_vocab) > 0:
            output_dirs = [os.path.split(fpath)[0] for fpath in filepaths]
            AnnotatedFeatureVocabBuilder.save(split_type=None,
                                file_dir=output_dirs,
                                save_key=key,
                                vocab=m_vocab["freqs"],
                                specials=m_vocab["specials"],
                                logger=logger,
                                ext=ext)


class AnnotatedPairing():
    def __init__(self):
        super().__init__()
        self.pos_vocab = None
        self.inarc_vocab = None
        self.subword_depth_stat = None
        self.coref_type_vocab = None
        self.coref_number_vocab = None
        self.coref_animacy_vocab = None
        self.coref_gender_vocab = None

    def __call__(
        self,
        config,
        build_vocab,
        use_source_corefs,
        use_target_corefs,
        use_entity_links,
        logger
    ):
        if build_vocab:
            self.pos_vocab = AnnotatedFeatureVocabBuilder(config.vocab_specials)
            self.inarc_vocab = AnnotatedFeatureVocabBuilder(config.vocab_specials)
            self.subword_depth_stat = AnnotatedNumberFeatureStat()
            if use_source_corefs or use_target_corefs:
                self.coref_type_vocab = AnnotatedFeatureVocabBuilder(config.vocab_specials)
                self.coref_number_vocab = AnnotatedFeatureVocabBuilder(config.vocab_specials)
                self.coref_animacy_vocab = AnnotatedFeatureVocabBuilder(config.vocab_specials)
                self.coref_gender_vocab = AnnotatedFeatureVocabBuilder(config.vocab_specials)

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
            self.process(config=config,
                         split_type=split_type,
                         atcl_files=atcl_files,
                         hlit_files=hlit_files,
                         tokenizer=tokenizer,
                         use_source_corefs=use_source_corefs,
                         use_target_corefs=use_target_corefs,
                         logger=logger)
        if self.pos_vocab:
            self.pos_vocab.save(split_type=split_type,
                                file_dir=config.output_dir,
                                save_key=config.vocabs["pos.vocab"],
                                vocab=self.pos_vocab.vocab,
                                specials=self.pos_vocab.specials,
                                logger=logger,
                                ext="")
        if self.inarc_vocab:
            self.inarc_vocab.save(split_type=split_type,
                                file_dir=config.output_dir,
                                save_key=config.vocabs["inarc.vocab"],
                                vocab=self.inarc_vocab.vocab,
                                specials=self.inarc_vocab.specials,
                                logger=logger,
                                ext="")
        if self.coref_type_vocab:
            self.coref_type_vocab.save(split_type=split_type,
                                file_dir=config.output_dir,
                                save_key=config.vocabs["coref.type.vocab"],
                                vocab=self.coref_type_vocab.vocab,
                                specials=self.coref_type_vocab.specials,
                                logger=logger,
                                ext="")
        if self.coref_number_vocab:
            self.coref_number_vocab.save(split_type=split_type,
                                file_dir=config.output_dir,
                                save_key=config.vocabs["coref.number.vocab"],
                                vocab=self.coref_number_vocab.vocab,
                                specials=self.coref_number_vocab.specials,
                                logger=logger,
                                ext="")
        if self.coref_animacy_vocab:
            self.coref_animacy_vocab.save(split_type=split_type,
                                file_dir=config.output_dir,
                                save_key=config.vocabs["coref.animacy.vocab"],
                                vocab=self.coref_animacy_vocab.vocab,
                                specials=self.coref_animacy_vocab.specials,
                                logger=logger,
                                ext="")
        if self.coref_gender_vocab:
            self.coref_gender_vocab.save(split_type=split_type,
                                file_dir=config.output_dir,
                                save_key=config.vocabs["coref.gender.vocab"],
                                vocab=self.coref_gender_vocab.vocab,
                                specials=self.coref_gender_vocab.specials,
                                logger=logger,
                                ext="")
        if self.subword_depth_stat:
            AnnotatedNumberFeatureStat.save(file_dir=config.output_dir,
                                            name="subword.depth.stat",
                                            save_map={"size": self.subword_depth_stat.size,
                                                      "padding_id": config.subword_depth_padding_id,
                                                      "freq": self.subword_depth_stat.stat},
                                            logger=logger)

    def process(
        self,
        config,
        split_type,
        atcl_files,
        hlit_files,
        tokenizer,
        use_source_corefs,
        use_target_corefs,
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
        atcl_afc = AnnotatedFeatureComposer(
                        config,
                        tokenizer,
                        atcl_aws_masker,
                        use_source_corefs,
                        use_ner=use_entity_links
                    )
        hlit_aws_masker = AnnotatedWordSegmentMasker()
        hlit_afc = AnnotatedFeatureComposer(
                        config,
                        tokenizer,
                        hlit_aws_masker,
                        use_target_corefs,
                        use_ner=use_entity_links
                    )

        error_lines = []
        composed = {"atcl": [], "hlit": []}
        save_ids = {"atcl": atcl_files.output,
                    "hlit": hlit_files.output}
        with ChunkSaver(save_ids, config.chunk_size, convert_json=False) as saver, \
            open(atcl_files.annotation, "r", encoding="utf-8") as atcl_fp, \
            open(hlit_files.annotation, "r", encoding="utf-8") as hlit_fp, \
            open(downloaded_path, "r", encoding="utf-8") as doc_fp:
            # debugging_iterations = 4
            for index, (atcl_line, hlit_line) in enumerate(zip(atcl_fp, hlit_fp)):
                # if index >= debugging_iterations:
                #     break
                try:
                    (atcl_doc, hlit_doc) = next(doc_iter(doc_fp))
                    atcl_data = atcl_afc(atcl_line, atcl_doc, logger)
                    hlit_data = hlit_afc(hlit_line, hlit_doc, logger)
                except Exception as ex:
                    error_lines.append(index)
                    logger.error(f"{str(ex)} @sample {index}.")
                    continue

                if self.pos_vocab:
                    self.pos_vocab.vocab = [atcl_data["pos"], hlit_data["pos"]]
                if self.inarc_vocab:
                    self.inarc_vocab.vocab = [atcl_data["inarc"], hlit_data["inarc"]]
                if self.coref_type_vocab:
                    if use_source_corefs:
                        self.coref_type_vocab.vocab = [atcl_data["coref_type"]]
                    if use_target_corefs:
                        self.coref_type_vocab.vocab = [hlit_data["coref_type"]]
                if self.coref_number_vocab:
                    if use_source_corefs:
                        self.coref_number_vocab.vocab = [atcl_data["coref_number"]]
                    if use_target_corefs:
                        self.coref_number_vocab.vocab = [hlit_data["coref_number"]]
                if self.coref_animacy_vocab:
                    if use_source_corefs:
                        self.coref_animacy_vocab.vocab = [atcl_data["coref_animacy"]]
                    if use_target_corefs:
                        self.coref_animacy_vocab.vocab = [hlit_data["coref_animacy"]]
                if self.coref_gender_vocab:
                    if use_source_corefs:
                        self.coref_gender_vocab.vocab = [atcl_data["coref_gender"]]
                    if use_target_corefs:
                        self.coref_gender_vocab.vocab = [hlit_data["coref_gender"]]
                if self.subword_depth_stat:
                    self.subword_depth_stat.stat = [atcl_data["subword_depth"],
                                                    hlit_data["subword_depth"]]

                ok = config.max_len == -1 or \
                    (len(atcl_data["input_ids"]) <= config.max_len and \
                     len(hlit_data["input_ids"]) <= config.max_len)
                if not ok:
                    error_lines.append(index)
                    logger.error(f"Exceed max length @sample {index}.")
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
                    logger.error(f"{str(ex)} @sample {index}.")
                    continue

                if saver(features=composed, index=index, last_save=False):
                    [composed[k].clear() for k, v in composed.items()]
            # If any remaining
            if saver(features=composed, index=index, last_save=True):
                [composed[k].clear() for k, v in composed.items()]

        if len(error_lines):
            logger.warning(f"{split_type} has {len(error_lines)} improperly parsed articles of indexed list {error_lines}")

    def get_source_files(self, split_type, pair_type, src_dir, output_dir,
                         source_stem, output_stem):
        source_file = f"{split_type}.{pair_type}{source_stem}"
        source_filepath = os.path.join(src_dir, source_file)
        output_filepath = os.path.join(output_dir, f"{split_type}.{pair_type}{output_stem}")
        return ModelOutput({"annotation": source_filepath,
                            "output": output_filepath})


class FeatureType2Id():
    '''
        Convert pos to pos id by pos vocab.
    '''
    def __call__(
        self,
        config,
        use_source_corefs,
        use_target_corefs,
        logger
    ):
        use_corefs = use_source_corefs or use_target_corefs
        vocabs = {**config.vocabs, **config.coref_vocabs} if use_corefs else config.vocabs
        vocabs = Vocabase.load(config.src_dir, vocabs)
        for split_type in config.split_types:
            for pair_name, pair_type in config.pair_types.items():
                src_filepath = os.path.join(config.src_dir, f"{split_type}.{pair_type}{config.input_stem}")
                output_filepath = os.path.join(config.output_dir, f"{split_type}.{pair_type}{config.output_stem}")

                dataset = []
                save_id = "stype2id"
                with ChunkSaver({save_id: output_filepath}, config.chunk_size) as saver, \
                    open(src_filepath, "r", encoding="utf-8") as fp:
                    for index, line in enumerate(fp):
                        data = eval(line.strip())
                        for key, vocab in vocabs.items():
                            key = key.replace(".vocab", "")
                            key = key.replace(".", "_") if "." in key else key
                            if key in data:
                                new_data = self.handle_nested(data[key], vocab, config.unk_token)
                                data[key] = new_data
                            else:
                                logger.warning(f"!!!{pair_name} does not have data keyed by {key}!!!")
                        dataset.append(data)

                        if saver(features={save_id: dataset}, index=index, last_save=False):
                            dataset = []

                    # If any remaining
                    if saver(features={save_id: dataset}, index=index, last_save=True):
                        dataset = []

    def handle_nested(self, data, vocab, unk_token):
        new_data = []
        for item in data:
            if isinstance(item, list):
                d = self.handle_nested(item, vocab, unk_token) # Recursive
                new_data.append(d)
            else:
                try:
                    id = vocab[item]
                except:
                    id = vocab[unk_token]
                new_data.append(id)
        return new_data



def nullstr2none(v):
    return None if v.lower() in ("none", "null") else v

from parse_args import TrainDataArgsParse
def main():
    import logging
    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger(__name__)

    args = TrainDataArgsParse()()

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
    if args.build_compose:
        logger.info("Composing annotation into dataset...")
        cfg = ModelOutput(json.loads(args.compose))
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
        if args.build_vocab is not None:
            cfg.build_vocab = args.build_vocab
        logger.info(f"config: {cfg}")
        pairing = AnnotatedPairing()
        pairing(cfg,
                args.build_vocab,
                use_source_corefs=args.use_source_corefs,
                use_target_corefs=args.use_target_corefs,
                logger=logger)

    if args.reconcile_vocab:
        logger.info("Reconciling structure data label vocabulary...")
        cfg = ModelOutput(json.loads(args.compose))
        if args.output_dir is not None:
            cfg.src_dir = args.output_dir
        if args.output_dir is not None:
            cfg.output_dir = args.output_dir
        reconciler = VocabReconciler()
        reconciler(cfg, logger=logger)

    # Replace feature values by vocab ids
    if args.build_stype2id:
        logger.info("Converting structure label into id...")
        cfg = ModelOutput(json.loads(args.stype2id))
        if args.output_dir is not None:
            cfg.src_dir = args.output_dir
        if args.output_dir is not None:
            cfg.output_dir = args.output_dir
        if split_types is not None:
            cfg.split_types = split_types
        if pair_types is not None:
            cfg.pair_types = pair_types
        stype2id = FeatureType2Id()
        stype2id(cfg,
                 use_source_corefs=args.use_source_corefs,
                 use_target_corefs=args.use_target_corefs,
                 logger=logger)

    logger.info("Done!")

if __name__ == "__main__":
    main()
