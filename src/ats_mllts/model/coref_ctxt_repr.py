from __future__ import unicode_literals, print_function, division
from typing import Optional
import torch
from model.datautil import get_unique_index
from model.coref_ctxt_encoder import CnnModel
from model.er_modeling.representation_models import RepresentationModel


def restore_assign(data, sorted_indices, uniq_loc, uniq_inv_idx):
    _, counts = torch.unique_consecutive(uniq_inv_idx, return_counts=True)
    # values = data[uniq_loc].repeat_interleave(counts)
    values = data.repeat_interleave(counts, dim=0)
    # sorted_indices contains the after-sorted locations of the original data.
    # Resorting sorted_indices in ascedent order creates an one-to-one mapping
    # between the original data indices (implied indexing order) and
    # the after-sorted indices of the original data (i.e. sorted_indices).
    # Thus, use the restored indices to slice the data assigns data to proper locations
    # matching the original data.
    _, restore_indices = torch.sort(sorted_indices, dim=0, descending=False, stable=True)
    output = values[restore_indices.view(-1)]
    return output


class MentionContextRepresentationModel(RepresentationModel):
    def __init__(self, n_channels, kernel_size, output_size_factor, max_length):
        super().__init__()
        self.repr_model = CnnModel(n_channels, kernel_size, output_size_factor, max_length)
        self.max_length = max_length

    def forward(
        self,
        indices: Optional[torch.LongTensor],
        initial_states: torch.FloatTensor = None,
        **kwargs
    ):
        '''
            indices: Coreference mention indices.
            initial_states: token sequence states.
        '''
        h = indices
        x = initial_states
        # Get referent-context(spans) map.
        contexts = kwargs.pop("contexts", None)

        # Avoid compute context representation for the repeated index.
        sorted_values, sorted_indices = torch.sort(h[:,None], dim=0, descending=False, stable=True)
        uniq_loc, uniq_inv_idx = get_unique_index(sorted_values, sorted_indices)
        h_indices = h[uniq_loc]

        # Gather the initial latent states of the context.
        gathered_inputs, _ = \
            self.gather_context_inputs(h_indices, contexts, x.view(-1, x.shape[-1]), self.max_length)
        # Learning context representation.
        output = self.repr_model(gathered_inputs)
        output = output.squeeze(dim=1)
        # Assign the output to match the correct order and duplicates of input indices.
        cr = restore_assign(data=output,
                            sorted_indices=sorted_indices,
                            uniq_loc=uniq_loc,
                            uniq_inv_idx=uniq_inv_idx)
        return cr

    def gather_context_inputs(self, h_indices, contexts, inputs, max_length):
        '''
            Use antecedent as batch dimension.
        '''
        counts = []
        ctxt_inputs = []
        for i, antec_index in enumerate(h_indices.tolist()):
            indices = torch.zeros((0), dtype=h_indices.dtype, device=h_indices.device)
            for span in contexts[antec_index]:
                # Expand span range into index sequence for referent-references.
                indices = torch.cat((indices,
                                     torch.arange(*span, dtype=indices.dtype, device=indices.device)),
                                    dim=0)
            counts.append(len(indices))
            # Gather the indexed latent states.
            x = inputs[indices]
            # Padding.
            nl, nd = x.size()
            n_pad = max_length - nl
            x = torch.cat((x, torch.zeros((n_pad, nd), device=inputs.device)), dim=0)
            ctxt_inputs.append(x)

        # batch up
        ctxt_inputs = torch.stack(ctxt_inputs, dim=0)
        return (ctxt_inputs, counts)


class ContextRepresentationModel(RepresentationModel):
    def __init__(self, n_channels, kernel_size, output_size_factor, max_length):
        super().__init__()
        self.repr_model = CnnModel(n_channels, kernel_size, output_size_factor, max_length)
        self.max_length = max_length

    def forward(
        self,
        indices: Optional[torch.LongTensor],
        initial_states: torch.FloatTensor = None,
        **kwargs
    ):
        '''
            indices: The tensor of context spans.
            initial_states: token sequence states.
        '''
        x = initial_states

        # Avoid compute context representation for the repeated spans.
        h_indices, uniq_inv_idx = torch.unique(indices, sorted=True, return_inverse=True, return_counts=False, dim=0)

        # Gather the initial latent states of the context.
        gathered_inputs, _ = \
            self.gather_context_inputs(h_indices, x.view(-1, x.shape[-1]), self.max_length)
        # Learning context representation.
        output = self.repr_model(gathered_inputs)
        output = output.squeeze(dim=1)
        # Assign the output back to the input context spans.
        cr = output[uniq_inv_idx]
        return cr

    def gather_context_inputs(self, h_indices, inputs, max_length):
        counts = []
        ctxt_inputs = []
        for i, span in enumerate(h_indices):
            # Expand span range into index sequence for antecedent-references.
            indices = torch.arange(*span, dtype=h_indices.dtype, device=h_indices.device)
            counts.append(len(indices))
            # Gather the indexed latent states.
            x = inputs[indices]
            # Padding.
            nl, nd = x.size()
            n_pad = max_length - nl
            x = torch.cat((x, torch.zeros((n_pad, nd), device=inputs.device)), dim=0)
            ctxt_inputs.append(x)

        # batch up
        ctxt_inputs = torch.stack(ctxt_inputs, dim=0)
        return (ctxt_inputs, counts)
