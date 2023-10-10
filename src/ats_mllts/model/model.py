from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from transformers import AutoConfig, CONFIG_MAPPING, BartConfig
from transformers import BartForConditionalGeneration
from model.datautil import (
    mask_out_special,
    flat_gather,
    sparsify_values,
    sparsify_indices,
    fill_root_head_loc,
    densify_values
)
from model.subword_net import SubwordNet
from model.coref_mrl import CorefNet
from model.factum import Factum
from model.semantic_similarity import SemanticSimilarityMeasure
from model.tag_regularizer import TagRegularizer


sanity_check=False  # Turn on for debugging sanity check

class Model(nn.Module):
    def __init__(self, args, options, vocabs, logger):
        super().__init__()

        # Instantiate baseline module
        if args.base_model_pretrained_name is not None:
            logger.info("Creating seq2seq model from pretrained weights.")
            self.seq2seq = BartForConditionalGeneration.from_pretrained(args.base_model_pretrained_name)
        elif args.base_model_config_name:
            config = AutoConfig.from_pretrained(args.base_model_config_name)
            logger.info("Creating seq2seq model from scratch using pretrained configuration.")
            self.seq2seq = BartForConditionalGeneration(config)
        elif options.base_model:
            logger.info("Creating seq2seq model from configuration.")
            config = options.base_model.to_dict()
            self.seq2seq = BartForConditionalGeneration(BartConfig(**config))
        else:
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            logger.info("Creating seq2seq model from scratch.")
            self.seq2seq = BartForConditionalGeneration(config)

        # Auxiliary module config
        if "subword" not in options.aux_model.exclude_modules:
            self.subword_net = SubwordNet(options.aux_model.subword)

        if "enctag_classifier" not in options.aux_model.exclude_modules:
            self.enctag_regularizer = TagRegularizer(
                                          options.aux_model.enctag_classifier,
                                          vocabs["pos.vocab.json"]
                                      )

        if "dectag_classifier" not in options.aux_model.exclude_modules:
            self.dectag_regularizer = TagRegularizer(
                                         options.aux_model.dectag_classifier,
                                         vocabs["pos.vocab.json"]
                                     )

        if "coref_mrl" not in options.aux_model.exclude_modules:
            self.coref_mrl = CorefNet(options.aux_model.coref_mrl,
                                    attr_map=options.aux_model.struct["attr_map"],
                                    type_vocab=vocabs["coref.type.vocab.json"],
                                    animacy_vocab=vocabs["coref.animacy.vocab.json"],
                                    number_vocab=vocabs["coref.number.vocab.json"],
                                    gender_vocab=vocabs["coref.gender.vocab.json"],
                                    logger=logger)

        # Fact coherence net
        if "factum" not in options.aux_model.exclude_modules:
            self.factum = Factum(options.aux_model.factum,
                                 struct_opt=options.aux_model.struct,
                                 feat_vocab=vocabs["inarc.vocab.json"],
                                 inarc_embedding_shared=None,
                                 logger=logger)

            self.sscost = SemanticSimilarityMeasure(options.aux_model.factum)

    def forward(self, batch, options, iepoch):
        inputs = batch[0]
        struct_inputs = batch[1]
        struct_labels = batch[2]

        outputs = self.seq2seq(**inputs,
                            output_attentions=True,
                            output_hidden_states=True)

        m_output = {}
        m_output["cost"] = outputs.loss.cpu()

        if "subword" not in options.aux_model.exclude_modules:
            h_x = self.subword_net({"data": outputs.encoder_hidden_states[-1],
                                    "attention_mask": inputs["attention_mask"],
                                    "token_head": struct_inputs["subword_edge"],
                                    "token_depth": struct_inputs["subword_depth"],
                                    "token_sparse_mask": struct_inputs["subword_mask"],
                                    "token_dense_mask": struct_inputs["subword_mask_mask"]})

            h_y = self.subword_net({"data": outputs.decoder_hidden_states[-1],
                                    "attention_mask": inputs["decoder_attention_mask"],
                                    "token_head": struct_labels["subword_edge"],
                                    "token_depth": struct_labels["subword_depth"],
                                    "token_sparse_mask": struct_labels["subword_mask"],
                                    "token_dense_mask": struct_labels["subword_mask_mask"]})
        else:
            h_x = outputs.encoder_hidden_states[-1]
            h_y = outputs.decoder_hidden_states[-1]

        # Encoder input POS tag classification
        if "enctag_classifier" not in options.aux_model.exclude_modules:
            x_tag_cost, x_tag_acc = self.enctag_regularizer(
                                        latent_states=h_x,
                                        attention_mask=inputs["attention_mask"],
                                        token_mask=struct_inputs["token_mask"],
                                        token_mask_mask=struct_inputs["token_mask_mask"],
                                        pos=struct_inputs["pos"],
                                        pos_mask=struct_inputs["pos_mask"])
            m_output["cost"] += x_tag_cost.cpu()
            m_output["x_tag_acc"] = x_tag_acc.cpu()

        if "dectag_classifier" not in options.aux_model.exclude_modules:
            y_tag_cost, y_tag_acc = self.dectag_regularizer(
                                        latent_states=h_y,
                                        attention_mask=inputs["decoder_attention_mask"],
                                        token_mask=struct_labels["token_mask"],
                                        token_mask_mask=struct_labels["token_mask_mask"],
                                        pos=struct_labels["pos"],
                                        pos_mask=struct_labels["pos_mask"])
            m_output["cost"] += y_tag_cost.cpu()
            m_output["y_tag_acc"] = y_tag_acc.cpu()

        if "coref_mrl" not in options.aux_model.exclude_modules \
            and options.training.warmup.coref_mrl <= iepoch <= options.training.cooldown.coref_mrl:
            (coref_cost, _) = self.coref_mrl((h_x, struct_inputs))
            m_output["cost"] += coref_cost.cpu()

        # Factum regularisation
        if "factum" not in options.aux_model.exclude_modules and \
            options.training.warmup["factum"] <= iepoch:
            ignore_value = -100
            # Heads are indices
            src_token_head, src_token_dense_mask, src_head_mask_value = \
                self.sparsify_indices(token_mask=struct_inputs["token_mask"],
                                    token_mask_mask=struct_inputs["token_mask_mask"],
                                    indices=struct_inputs["head"],
                                    indices_mask=struct_inputs["head_mask"],
                                    ignore_value=ignore_value)
            src_sparse_head_mask = struct_inputs["token_mask"]
            if not options.aux_model.factum.head_self_loop:
                (fill_index, src_token_head) = fill_root_head_loc(
                                            token_mask=struct_inputs["token_mask"],
                                            token_mask_mask=struct_inputs["token_mask_mask"],
                                            root_head_mask=struct_inputs["root_head_mask"],
                                            root_head_mask_mask=struct_inputs["root_head_mask_mask"],
                                            sparse_head=src_token_head,
                                            fill_value=ignore_value)
                shape = src_sparse_head_mask.shape
                src_sparse_head_mask = src_sparse_head_mask.reshape(-1) \
                                                            .index_fill_(dim=0, index=fill_index, value=0) \
                                                            .reshape(shape)

            src_token_inarc, src_token_inarc_dense_mask, src_inarc_mask_value = \
                self.sparsify_values(token_mask=struct_inputs["token_mask"],
                                    token_mask_mask=struct_inputs["token_mask_mask"],
                                    value=struct_inputs["inarc"],
                                    value_mask=struct_inputs["inarc_mask"],
                                    ignore_value=ignore_value)
            src_token_depth, src_token_depth_dense_mask, src_depth_mask_value = \
                self.sparsify_values(token_mask=struct_inputs["token_mask"],
                                    token_mask_mask=struct_inputs["token_mask_mask"],
                                    value=struct_inputs["depth_d"],
                                    value_mask=struct_inputs["depth_d_mask"],
                                    ignore_value=ignore_value)
            if sanity_check:
                assert torch.all(src_token_dense_mask.eq(src_token_inarc_dense_mask))
                assert torch.all(src_token_dense_mask.eq(src_token_depth_dense_mask))
                assert torch.all(src_token_inarc_dense_mask.eq(src_token_depth_dense_mask))
                _, sparsed_mask = mask_out_special(dense_mask=struct_inputs["token_mask_mask"],
                                                        sparse_mask=struct_inputs["token_mask"],
                                                        left_n=1, right_n=1)
                assert torch.all(sparsed_mask.eq((src_head_mask_value>0).long()))
                assert torch.all(sparsed_mask.eq((src_inarc_mask_value>0).long()))
                assert torch.all(sparsed_mask.eq((src_depth_mask_value>0).long()))

            # Labels
            tgt_token_head, tgt_token_dense_mask, _ = \
                self.sparsify_indices(token_mask=struct_labels["token_mask"],
                                    token_mask_mask=struct_labels["token_mask_mask"],
                                    indices=struct_labels["head"],
                                    indices_mask=struct_labels["head_mask"],
                                    ignore_value=ignore_value)
            tgt_sparse_head_mask = struct_labels["token_mask"]
            if not options.aux_model.factum.head_self_loop:
                (fill_index, tgt_token_head) = fill_root_head_loc(
                                            token_mask=struct_labels["token_mask"],
                                            token_mask_mask=struct_labels["token_mask_mask"],
                                            root_head_mask=struct_labels["root_head_mask"],
                                            root_head_mask_mask=struct_labels["root_head_mask_mask"],
                                            sparse_head=tgt_token_head,
                                            fill_value=ignore_value)
                shape = tgt_sparse_head_mask.shape
                tgt_sparse_head_mask = tgt_sparse_head_mask.reshape(-1) \
                                                            .index_fill_(dim=0, index=fill_index, value=0) \
                                                            .reshape(shape)

            tgt_token_inarc, _, _ = \
                self.sparsify_values(token_mask=struct_labels["token_mask"],
                                    token_mask_mask=struct_labels["token_mask_mask"],
                                    value=struct_labels["inarc"],
                                    value_mask=struct_labels["inarc_mask"],
                                    ignore_value=ignore_value)
            tgt_token_depth, _, _ = \
                self.sparsify_values(token_mask=struct_labels["token_mask"],
                                    token_mask_mask=struct_labels["token_mask_mask"],
                                    value=struct_labels["depth_d"],
                                    value_mask=struct_labels["depth_d_mask"],
                                    ignore_value=ignore_value)

            factum_inputs = {
                        # source structure
                        "h_x": h_x,
                        "x_mask": inputs["attention_mask"],
                        "src_token_arc_head": src_token_head,
                        "src_token_inarc_type": src_token_inarc,
                        "src_token_depth": src_token_depth,
                        "src_token_sparse_mask": src_sparse_head_mask,
                        "src_token_dense_mask": src_token_dense_mask,
                        "src_word_inarc_type": struct_inputs["inarc"],
                        "src_word_inarc_type_mask": struct_inputs["inarc_mask"],
                        # target structure
                        "h_y": h_y,
                        "y_mask": inputs["decoder_attention_mask"],
                        "tgt_token_arc_head": tgt_token_head,
                        "tgt_token_inarc_type": tgt_token_inarc,
                        "tgt_token_depth": tgt_token_depth,
                        "tgt_token_sparse_mask": tgt_sparse_head_mask,
                        "tgt_token_dense_mask": tgt_token_dense_mask,
                        "tgt_word_inarc_type": struct_labels["inarc"],
                        "tgt_word_inarc_type_mask": struct_labels["inarc_mask"],
                        }
            (h_x, h_y) = self.factum(factum_inputs)

            # Compute cost
            src_word_inarc_type_mask = struct_inputs["inarc_mask"]
            tgt_word_inarc_type_mask = struct_labels["inarc_mask"]
            x_repr = densify_values(h_x, src_sparse_head_mask*src_token_dense_mask)
            y_repr = densify_values(h_y, tgt_sparse_head_mask*tgt_token_dense_mask)

            ss_cost = self.sscost(
                x_repr,
                y_repr,
                src_word_inarc_type_mask,
                tgt_word_inarc_type_mask
            )
            m_output["cost"] += ss_cost.cpu()

        return m_output

    def sparsify_indices(self, token_mask, token_mask_mask,
                        indices, indices_mask, ignore_value):
        '''
            Map the indices at word level to the corresponding token levels
            for the sparse token encoding e.g. byte-pair encoding.
        '''
        # Mask out BOS and EOS that are not part of structure indices.
        sparsed_mask_mask, _ = mask_out_special(dense_mask=token_mask_mask,
                                                sparse_mask=None,
                                                left_n=1, right_n=1)

        sparsed_indices = sparsify_indices(token_mask, sparsed_mask_mask,
                                           indices, indices_mask, ignore_value)
        sparsed_mask_value = None
        if sanity_check: # Debugging
            mask_value_gathered = flat_gather(indices_mask, indices_mask)
            sparsed_mask_value = sparsify_values(token_mask*sparsed_mask_mask,
                                                 mask_value_gathered,
                                                 init_val=ignore_value)

        return sparsed_indices, sparsed_mask_mask, sparsed_mask_value

    def sparsify_values(
        self,
        token_mask,
        token_mask_mask,
        value,
        value_mask,
        ignore_value=-100,
        bos_eos_padded=True
    ):
        '''
            Assign the values indexed at word level to the corresponding token levels
            for the sparse token encoding e.g. byte-pair encoding.

            bos_eos_padded: BOS and EOS special tokens have been padded to token sequence if True.
        '''
        # Mask out BOS and EOS that are not part of structure indices.
        if bos_eos_padded:
            sparsed_mask_mask, _ = mask_out_special(dense_mask=token_mask_mask,
                                                    sparse_mask=None,
                                                    left_n=1, right_n=1)
        else:
            sparsed_mask_mask = token_mask_mask

        token_mask = token_mask*sparsed_mask_mask
        if sanity_check:
            token_mask_sum = torch.sum(token_mask, dim=-1)
            value_mask_sum = torch.sum(value_mask, dim=-1)
            # delta = token_mask_sum - value_mask_sum
            assert torch.all(token_mask_sum.eq(value_mask_sum))

        # Usee token mask and value mask to sparsify values onto (first) token/subtoken positions.
        value_gathered = flat_gather(value, value_mask)
        sparsed_value = sparsify_values(token_mask, value_gathered, init_val=ignore_value)

        sparsed_mask_value = None
        if sanity_check: # Debugging
            mask_value_gathered = flat_gather(value_mask, value_mask)
            sparsed_mask_value = sparsify_values(token_mask, mask_value_gathered, init_val=ignore_value)

        return sparsed_value, sparsed_mask_mask, sparsed_mask_value

    @torch.no_grad()
    def generate(
        self,
        batch,
        options,
        **model_kwargs,
    ):
        inputs = batch[0]

        return self.seq2seq.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            **model_kwargs)
