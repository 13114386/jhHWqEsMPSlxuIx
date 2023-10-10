from __future__ import unicode_literals, print_function, division
import torch.nn as nn
# from pykeen.losses import MarginRankingLoss, SoftMarginRankingLoss
from model.er_modeling.losses import MarginRankingLoss
from model.coref_mrl_sampler import (
    CorefNegativeSampler,
    CorefCtxtNegativeSampler,
)
from model.er_modeling.complex import ComplEx


class CorefCtxtMRL(nn.Module):
    def __init__(
        self,
        er_model,
        rr_model,
        cr_model,
        ar_model,
        loss_margin,
        lambda_w,
        logger,
        **kwargs
    ):
        '''
            Referent-context(spans) MRL.
        '''
        super().__init__()
        self.negative_sampler = CorefCtxtNegativeSampler()
        self.model = ComplEx(referent_repr_model=er_model,
                            relation_repr_model=rr_model,
                            reference_repr_model=cr_model,
                            attrib_repr_model=ar_model,
                            **kwargs)
        self.loss = MarginRankingLoss(margin=loss_margin)
        self.lambda_w = lambda_w

    def forward(self, inputs):
        '''
            Note:
                Negative sampler returns batch flattened samples
                if batch_offsets is not None.
                It makes computational efficiency of scoring.
        '''
        h_x = inputs["h_x"]
        batch_triples = inputs["triples"]
        batch_offsets = inputs["offsets"]
        batch_contexts = inputs["contexts"]
        batch_entity_attribs = inputs["attributes"]
        mr_samples = self.negative_sampler(
                    batch_triples,
                    batch_offsets,
                    batch_contexts,
                    batch_entity_attribs
                )
        referents = mr_samples["referents"].squeeze(dim=-1)
        positive_scores = self.model.score_hrt(
                                h_x,
                                hs=referents,
                                rs=None,
                                ts=mr_samples["positive_samples"],
                                attribs=(mr_samples["ref_attribs"], None),
                                mode=None
                            )
        negative_scores = self.model.score_hrt(
                                h_x,
                                hs=referents,
                                rs=None,
                                ts=mr_samples["negative_samples"],
                                attribs=(mr_samples["ref_attribs"], None),
                                mode=None
                            )

        loss = self.loss.process_slcwa_scores(
                    positive_scores=positive_scores,
                    negative_scores=negative_scores,
                    label_smoothing=None,
                )
        loss = loss*self.lambda_w
        return (loss, h_x)


class CorefMRL(nn.Module):
    def __init__(
        self,
        er_model,
        rr_model,
        cr_model,
        ar_model,
        loss_margin,
        lambda_w,
        logger,
        **kwargs
    ):
        '''
            
            Referent-Reference MRL.
        '''
        super().__init__()
        self.negative_sampler = CorefNegativeSampler()
        self.model = ComplEx(referent_repr_model=er_model,
                            relation_repr_model=rr_model,
                            reference_repr_model=cr_model,
                            attrib_repr_model=ar_model,
                            **kwargs)
        self.loss = MarginRankingLoss(margin=loss_margin)
        self.lambda_w = lambda_w

    def forward(self, inputs):
        '''
            Note:
                Negative sampler returns batch flattened samples
                if batch_offsets is not None.
                It makes computational efficiency of scoring.
        '''
        h_x = inputs["h_x"]
        batch_triples = inputs["triples"]
        batch_offsets = inputs["offsets"]
        batch_contexts = inputs["contexts"]
        batch_attribs = inputs["attributes"]
        mr_samples = self.negative_sampler(
                        batch_triples,
                        batch_offsets,
                        batch_contexts,
                        batch_attribs
                    )

        positive_scores = self.model.score_hrt(
                                h_x,
                                hs=mr_samples["referents"],
                                rs=mr_samples["positve_relations"],
                                ts=mr_samples["positive_samples"],
                                attribs=(mr_samples["referent_attribs"],
                                         mr_samples["positive_attribs"]),
                                mode=None
                            )

        negative_scores = self.model.score_hrt(
                                h_x,
                                hs=mr_samples["referents"],
                                rs=mr_samples["negative_relations"],
                                ts=mr_samples["negative_samples"],
                                attribs=(mr_samples["referent_attribs"],
                                         mr_samples["negative_attribs"]),
                                mode=None
                            )

        loss = self.loss.process_slcwa_scores(
                    positive_scores=positive_scores,
                    negative_scores=negative_scores,
                    label_smoothing=None,
                )
        loss = loss*self.lambda_w
        return (loss, h_x)
