from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from model.classifier import POSTagClassifier
from model.cost import tag_cost, tag_acc
from model.datautil import (
    mask_out_special,
    flat_gather,
    sparsify_values,
)

sanity_check=False

class TagRegularizer(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        tag_vocab_size = len(vocab)
        # Consolidate vocab size from the vocab.
        config["n_class"] = tag_vocab_size
        self.classifier = POSTagClassifier(config)
        self.lambda_w = config["lambda"]

    def forward(
        self,
        latent_states,
        attention_mask,
        token_mask,
        token_mask_mask,
        tag,
        tag_mask
    ):
        x = self.classifier(latent_states)
        x_tag_cost, x_tag_acc = self.classify_tag(
                                    x=x,
                                    x_attn_mask=attention_mask,
                                    token_mask=token_mask,
                                    token_mask_mask=token_mask_mask,
                                    tag=tag,
                                    tag_mask=tag_mask
                                )
        return x_tag_cost*self.lambda_w, x_tag_acc

    def classify_tag(
        self,
        x,
        x_attn_mask,
        token_mask,
        token_mask_mask,
        tag,
        tag_mask,
        ignore_value=-100
    ):
        sparsed_tag, sparsed_mask_mask, _ = self.sparsify_values(
                                                token_mask,
                                                token_mask_mask,
                                                tag,
                                                tag_mask,
                                                ignore_value
                                            )

        if sanity_check:
            x_attn_mask_sum = torch.sum(x_attn_mask, dim=-1)
            assert torch.all(x_attn_mask_sum.eq(torch.sum(token_mask_mask, dim=-1)))

        # Cost and Acc
        x_tag_cost = tag_cost(x, sparsed_tag, ignore_index=ignore_value) # Cost
        x_tag_acc = tag_acc(x, sparsed_tag, sparsed_mask_mask) # Acc
        return x_tag_cost, x_tag_acc

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
