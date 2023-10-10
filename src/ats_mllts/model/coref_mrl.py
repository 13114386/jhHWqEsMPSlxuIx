from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from model.mlp import MLP
from model.er_modeling.representation_models import (
    IndexedStateRepresentationModel,
    AttributedStateRepresentationModel,
    EmbeddingRepresentationModel,
    PairedRelationRepresentationModel
)
from model.datautil import (
    get_unique_index,
    covert_word_to_token_indices
)
from model.coref_data_utils import (
    CorefGraphBuilder,
    EntityGraphBuilder,
    offset_key_to_head,
    get_indexed_attributes,
    uniquify_attributes,
    query_indexed_attributes
)
from model.coref_ctxt_repr import ContextRepresentationModel
from model.co_embedding import CoEmbedding
from model.coref_entity_gnn import EntityAggrNet
from model.coref_mrl_modules import (CorefCtxtMRL, CorefMRL)


class CorefNet(nn.Module):
    def __init__(
        self,
        config,
        attr_map,
        type_vocab,
        animacy_vocab,
        number_vocab,
        gender_vocab,
        logger
    ):
        super().__init__()
        self.coref_entity_builder = EntityGraphBuilder(config.ngram_bound) if config.aggregate_entity else None
        self.coref_entity_net = EntityAggrNet(config.entity_aggr) if config.aggregate_entity else None
        self.coref_graph_builder = CorefGraphBuilder()
        self.mlp = MLP([config.mlp.in_channels, config.mlp.out_channels],
                        activation=nn.ReLU())
        (er_m, rr_m, cr_m, ar_m) = \
            self._prepare_representation_models(
                                    config=config,
                                    attr_map=attr_map,
                                    type_vocab=type_vocab,
                                    animacy_vocab=animacy_vocab,
                                    number_vocab=number_vocab,
                                    gender_vocab=gender_vocab,
                                    logger=logger)

        kwargs = {"attributed": config.relational=="attributed"}
        self.ref_mrl = CorefMRL(er_model=er_m,
                                rr_model=rr_m,
                                cr_model=er_m,
                                ar_model=ar_m,
                                loss_margin=config.loss_margin,
                                lambda_w=config.lambda_w,
                                logger=logger,
                                **kwargs
                        )

        self.ctxt_mrl = CorefCtxtMRL(er_model=er_m,
                                    rr_model=None,
                                    cr_model=cr_m,
                                    ar_model=ar_m,
                                    loss_margin=config.loss_margin,
                                    lambda_w=config.lambda_w,
                                    logger=logger
                        ) if config.contextual else None

        self.aggregate_entity = config.aggregate_entity
        self.attributed = config.attributed
        self.contextual = config.contextual
        self.relational = config.relational
        self.mention_keyed = config.mention_keyed
        self.attribute_order = config.attribute_order

    def forward(self, inputs):
        '''
            Note:
                Negative sampler returns batch flattened samples
                if batch_offsets is not None.
                It makes computational efficiency of scoring.
        '''
        (h_x, struct_inputs) = inputs
        data = self.prepare_data(h_x, struct_inputs)
        h_x = data.pop("h_x")
        h_x = self.mlp(h_x)
        nb, nl, _ = h_x.size()
        batch_offsets = [nl*b for b in range(nb)]
        batch_triples, batch_contexts, batch_attribs = self.get_triples(data)

        kwargs = {"h_x": h_x,
                  "triples": batch_triples,
                  "offsets": batch_offsets,
                  "contexts": batch_contexts,
                  "attributes": batch_attribs}
        loss, h_x = self.ref_mrl.forward(kwargs)
        if self.ctxt_mrl is not None:
            ctxt_loss, _ = self.ctxt_mrl.forward(kwargs)
            loss = loss + ctxt_loss
        return (loss, h_x)

    def get_triples(self, inputs):
        batch_triples = []
        # Entity attributes
        batch_entity_attribs = [] if self.attributed else None
        # Contexts
        batch_contexts = inputs.get("context", None) if self.contextual else None
        for ib, mentions in enumerate(inputs["mentions"]):
            # Sort by antecedent index first
            mentions = sorted(mentions, key=lambda x: x[0].item())
            triples = []
            mantec_attribs = [] if self.attributed else None
            mcoref_attribs = [] if self.attributed else None
            for (antecedent, corefs) in mentions:
                # Remove index duplicated corefs.
                values, indices = torch.sort(corefs[:,None], dim=0, descending=False, stable=True)
                uniq_loc, _ = get_unique_index(values, indices)
                corefs = corefs[uniq_loc]

                distances = torch.abs(corefs - antecedent)
                n_coref = corefs.shape[0]
                antecedent = antecedent.expand(n_coref)
                triple = (antecedent, distances, corefs)
                triples.append(triple)
                # Gather sparse coref_type, coref_animacy, coref_gender, coref_number by mentions
                if self.attributed:
                    entity_attrbitues = inputs["attrbitues"]
                    batch_counts = inputs["batch_counts"]
                    antecedent_attribs = query_indexed_attributes(antecedent, entity_attrbitues[batch_counts==ib])
                    antecedent_attribs = antecedent_attribs.expand(n_coref, -1)
                    corefs_attribs = query_indexed_attributes(corefs, entity_attrbitues[batch_counts==ib])
                    mantec_attribs.append(antecedent_attribs)
                    mcoref_attribs.append(corefs_attribs)
            triples = torch.cat([torch.stack(t, dim=0).T for t in triples], dim=0)
            # Add to batch
            batch_triples.append(triples)
            if self.attributed:
                mantec_attribs = torch.cat(mantec_attribs, dim=0)
                mcoref_attribs = torch.cat(mcoref_attribs, dim=0)
                batch_entity_attribs.append((mantec_attribs, mcoref_attribs))
        return batch_triples, batch_contexts, batch_entity_attribs

    def _prepare_representation_models(
        self,
        config,
        attr_map,
        type_vocab,
        animacy_vocab,
        number_vocab,
        gender_vocab,
        logger
    ):
        if config.attributed and config.relational != "attributed":
            assert config.entity_repr.in_channels == \
                config.mlp.out_channels + attr_map.dim*len(config.attribute_order)
            er_m = AttributedStateRepresentationModel(
                        in_channels=config.entity_repr.in_channels,
                        out_channels=config.entity_repr.out_channels,
                        activation=config.entity_repr.activation,
                        batch_norm=config.entity_repr.batch_norm,
                        dropout=config.entity_repr.dropout)
        else:
            er_m = IndexedStateRepresentationModel()

        rr_m = None
        if config.relational == "attributed":
            rr_m = PairedRelationRepresentationModel(
                        in_channels=config.attributed_rel.in_channels,
                        out_channels=config.attributed_rel.out_channels,
                        activation=config.attributed_rel.activation,
                        batch_norm=config.attributed_rel.batch_norm,
                        dropout=config.attributed_rel.dropout)
        elif config.relational == "distance":
            rr_m = EmbeddingRepresentationModel(
                        n_embeddings=config.distance_rel.n_embeddings,
                        n_dims=config.distance_rel.n_dims,
                        padding_idx=config.distance_rel.padding_idx
                    )

        cr_m = ContextRepresentationModel(
                    n_channels=config.ctxt.n_channels,
                    kernel_size=config.ctxt.kernel_size,
                    output_size_factor=config.ctxt.output_size_factor,
                    max_length=config.ctxt.max_length
                ) if config.contextual else None

        ar_m = CoEmbedding(
                    attr_struct=attr_map,
                    type_vocab=type_vocab,
                    animacy_vocab=animacy_vocab,
                    number_vocab=number_vocab,
                    gender_vocab=gender_vocab,
                    key_order=config.attribute_order,
                    logger=logger
                ) if config.attributed else None
        return (er_m, rr_m, cr_m, ar_m)

    def prepare_data(self, h_x, struct_inputs):
        # Get an entity key/head word index offset w.r.t. the first word of the entity.
        keyhead_offset, keyhead_offset_mask = None, None
        if self.mention_keyed:
            keyhead_offset, keyhead_offset_mask = \
                offset_key_to_head(
                    coref_index=struct_inputs["coref_index"],
                    coref_index_mask=struct_inputs["coref_index_mask"],
                    coref_entity_span=struct_inputs["coref_entity_span"],
                    coref_entity_span_mask=struct_inputs["coref_entity_span_mask"],
                    coref_key_index=struct_inputs["coref_head_index"],
                    coref_key_index_mask=struct_inputs["coref_head_index_mask"],
                )

        # Map word level indices onto token level indices
        ignore_value=-100
        entity_token_indices, entity_token_mask = \
            covert_word_to_token_indices(token_mask=struct_inputs["token_mask"],
                                        token_mask_mask=struct_inputs["token_mask_mask"],
                                        indices=struct_inputs["coref_index"],
                                        indices_mask=struct_inputs["coref_entity_mask"],
                                        indices_mask_mask=struct_inputs["coref_entity_mask_mask"],
                                        token_span_count=struct_inputs["subword_span"],
                                        token_span_count_mask=struct_inputs["subword_span_mask"],
                                        indices_offset_n=1, # offset from BOS
                                        flat_batch=False,
                                        ignore_value=ignore_value)
        # Build edges for aggregating multiple word entities by GNN.
        if self.aggregate_entity:
            _, nl = entity_token_indices.size()
            assert nl == struct_inputs["coref_index"].shape[-1], \
                    "covert_word_to_token_indices has changed length dimension."
            entity_edges, _, (_, _, _, _, _, batch_counts) = \
                self.coref_entity_builder(
                    coref_index=entity_token_indices, #struct_inputs["coref_index"],
                    coref_index_mask=struct_inputs["coref_index_mask"],
                    coref_entity_mask=struct_inputs["coref_entity_mask"],
                    coref_entity_mask_mask=entity_token_mask, #struct_inputs["coref_entity_mask_mask"],
                    coref_entity_span=struct_inputs["coref_entity_span"],
                    coref_entity_span_mask=struct_inputs["coref_entity_span_mask"],
                    coref_key_offset=keyhead_offset,
                    coref_key_offset_mask=keyhead_offset_mask,
                    batch_offset_required=True,
                    lone_entity_included=False,
                    entity_head_self_loop=False,
                    dep_to_head_direction=True)
  
            # Aggregate multiple word entity to a single word representation by GNN.
            entity_distances = torch.abs(entity_edges[0] - entity_edges[1])
            h_x = self.coref_entity_net({"data": h_x,
                                        "edge": entity_edges,
                                        "edge_feature": entity_distances,
                                        })

        # Bundle entity indices and their attributes.
        attrbitues = None
        if self.attributed:
            attributes = [struct_inputs["coref_" + k] for k in self.attribute_order]
            # Create an indexed attributes as a map (sliceable tensor by matching index).
            indexed_attrbitues = \
                get_indexed_attributes(
                    attributes=attributes,
                    entity_head=None,
                    entity_index=entity_token_indices,
                    entity_mask=struct_inputs["coref_entity_mask"],
                    entity_mask_mask=struct_inputs["coref_entity_mask_mask"],
                    entity_span=struct_inputs["coref_entity_span"],
                    entity_span_mask=struct_inputs["coref_entity_span_mask"],
                    key_offset=keyhead_offset,
                    key_offset_mask=keyhead_offset_mask,
                    batch_offset_required=False
                )
            attrbitues, batch_counts = uniquify_attributes(indexed_attrbitues, batch_counts)

        # Gather antecedent-coreferent pairs.
        g_data = self.coref_graph_builder(
                    coref_index=entity_token_indices,
                    coref_sent_num=struct_inputs["coref_sent_num"],
                    coref_repr_mask=struct_inputs["coref_repr_mask"],
                    coref_repr_mask_mask=struct_inputs["coref_repr_mask_mask"],
                    coref_entity_mask=struct_inputs["coref_entity_mask"],
                    coref_sent_num_mask=struct_inputs["coref_sent_num_mask"],
                    token_sentence_sizes=struct_inputs["tokenized_sent_sizes"],
                    token_mask=struct_inputs["token_mask"],
                    token_mask_mask=struct_inputs["token_mask_mask"],
                    coref_key_offset=keyhead_offset,
                    coref_key_offset_mask=keyhead_offset_mask,
                    include_ante=True,
                    edge_builder=None,
                    return_tensor=False
                )
        return {"h_x": h_x,
                "attrbitues": attrbitues,
                "batch_counts": batch_counts,
                "context": g_data["context"],
                "mentions": g_data["mention"]}
