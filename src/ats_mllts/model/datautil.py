from __future__ import unicode_literals, print_function, division
import numpy as np
import torch

srng = np.random.RandomState(seed=20210919)

class DropoutDim():
    '''
        Dropout along a specified dimension
    '''
    def __call__(self, x, dim, p):
        ndim = len(x.shape)
        if dim < 0:
            dim = ndim + dim
        size = x.shape[:dim+1]
        mask = self.dropout_calc(size, p)
        mask = torch.tensor(mask, dtype=torch.float32, device=x.device)
        dims = list(range(ndim-dim-1))
        for _ in dims:
            mask = mask.unsqueeze(-1)
        x = x*mask
        return x

    def dropout_calc(self, size, rate = 0.0):
        mask = srng.binomial(n=1, p=(1-rate), size=list(size))
        return mask

def cumulative(values):
    '''
        Compute a list of integer values into cumulative sums.
    '''
    length = len(values)
    values = [sum(values[0:x:1]) for x in range(0, length+1)]
    return values

def mask2symmtrx(masks, batch_first=True, zero_diag=False):
    '''
    masks: [nL, nB] / [nB, nL]
    '''
    ns = len(masks.shape)
    assert ns <= 2
    if not batch_first and ns > 1:
        masks = masks.transpose(dim0=0, dim1=1)
    if ns == 1:
        masks = masks[None,...]

    symmtrx_2d = masks[...,None]*masks[:,None,:]
    symmtrx_2d = symmtrx_2d.permute(1,2,0)
    if zero_diag:
        _, nL = masks.size()
        symmtrx_2d = symmtrx_2d - torch.eye(nL,
                                            dtype=symmtrx_2d.dtype,
                                            device=symmtrx_2d.device).reshape((nL, nL, 1))
        symmtrx_2d[symmtrx_2d==-1] = 0
    return symmtrx_2d

def fill_ignore_value(x, mask, ignore_value=-100):
    ignores = (1. - mask) * ignore_value
    x_ = x*mask + ignores.long()
    x_ = x_.contiguous()
    return x_

def remove_end_padding(mask, last_n):
    slens = torch.sum(mask, dim=1, keepdim=True)
    slefts = slens - last_n
    slens = slens - 1  # zero-indexed
    index = torch.cat((slefts, slens), dim=1)
    val = torch.zeros_like(index)
    new_mask = mask.scatter(1, index, val)
    return new_mask

def mask_out_special(dense_mask, sparse_mask, left_n, right_n):
    '''
        Mask out the number of specified bits at both ends.
    '''
    new_mask = remove_end_padding(dense_mask, last_n=right_n) # Remove EOS mask
    if left_n > -1:
        new_mask[:,:left_n] = 0 # Remove BOS

    new_sparse_mask = sparse_mask*new_mask if sparse_mask is not None else None
    return new_mask, new_sparse_mask

def get_cumsum(mask, use_mask, prepad=None):
    '''
        x: [nb, nd, nl]
    '''
    if use_mask:
        lens_per_sample = torch.sum(mask, dim=1)
    else:
        lens_per_sample = [mask.shape[-1]]*mask.shape[0]
        lens_per_sample = torch.tensor(lens_per_sample, device=mask.device)
    cumsum = torch.cumsum(lens_per_sample, dim=0)
    if prepad is not None:
        cumsum = torch.cat((prepad,cumsum), dim=0)
    return cumsum

def index_to_gather(mask, sparse_mask=None, flat=True, as_tuple=False):
    '''
        mask: The dense mask corresponding to actual length.
        sparse_mask: The mask of some positions in the dense mask is of interest.
    '''
    which = mask if sparse_mask is None else sparse_mask
    if flat:
        index = torch.nonzero(which.reshape(-1), as_tuple=as_tuple)
        index = index[0] if as_tuple else index.reshape(-1)
    else:
        index = torch.nonzero(which, as_tuple=as_tuple)
    return index

def flat_gather(source, mask):
    '''
        Gather those masked values in a flattened dense form.
    '''
    src_index_flat = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    if len(source.shape) <= 2:
        src_gathered = torch.gather(source.reshape(-1), -1, src_index_flat)
    elif len(source.shape) == 3:
        nb,nl,nd = source.size()
        source = source.reshape(nb*nl, -1)
        src_gathered = source[src_index_flat]
    return src_gathered

def sparsify_values(source_mask, values, init_val=-100, keepdim=True):
    '''
        Assign values to the masked positions, and return a new tensor.
    '''
    src_indices_flat = torch.nonzero(source_mask.reshape(-1), as_tuple=False).reshape(-1)
    sparsed = torch.ones_like(source_mask.reshape(-1))*init_val
    sparsed = torch.scatter(sparsed, -1, src_indices_flat, values)
    if keepdim:
        sparsed = sparsed.reshape(source_mask.shape)
    return sparsed

from itertools import chain
def calculate_sparse_offset(sparse_mask):
    '''
        Create offsets (to be used for sparsifying gathered dense positions).

        sparse_mask: sth. like first token mask of a word from byte-pair encoding.
    '''
    # Get word-corresponding first token positions from token mask.
    batch_index, sparse_index = torch.nonzero(sparse_mask, as_tuple=True)

    # Count word occurrences in each sample in mini batch.
    lens_per_sample = torch.bincount(batch_index, weights=None, minlength=0)

    # Get cumulative sum from the counts
    cumsum = torch.cumsum(lens_per_sample, dim=0)
    prepad = torch.zeros((1), dtype=cumsum.dtype, device=cumsum.device)
    cumsum = torch.cat((prepad, cumsum), dim=0).tolist()

    # Create offsets (to be used for gathered dense positions)
    offsets = [[cumsum[i]] * (cumsum[i+1] - cumsum[i]) for i in range(len(cumsum)-1)]
    offsets = list(chain.from_iterable(offsets)) # Flatten the list
    offsets = torch.tensor(offsets, device=sparse_mask.device)
    return sparse_index, offsets

def sparsify_indices(sparse_mask, sparse_mask_mask,
                    indices, indices_mask, ignore_value):
    '''
        Map the indices at word level to the corresponding token levels
        for the sparse token encoding e.g. byte-pair encoding.

        sparse_mask: Padded sparse mask tensor.
                     Note that padding token is not necessarily zero.
        sparse_mask_mask: The mask of the padded sparse mask tensor.
        indices: Padded dense tensor.
        indices_mask: The mask of the padded dense tensor.
        ignore_value: Fill-in values for ignored positions.
    '''
    # Ensure padding are zero-ed out.
    sparse_mask = sparse_mask*sparse_mask_mask

    # Create offsets for gathered heads.
    sparse_index, offsets = calculate_sparse_offset(sparse_mask)

    # Get word positions from head.
    dense_positions = flat_gather(indices, indices_mask)

    # Add gathered head with offsets.
    dense_positions = dense_positions + offsets

    # Map offsetted head positions to sparse token positions.
    sparse_positions = sparse_index[dense_positions]
    sparsed_indices = sparsify_values(sparse_mask, sparse_positions,
                                    init_val=ignore_value, keepdim=True)
    return sparsed_indices

def sparsify_mask(
    token_mask,
    token_mask_mask,
    word_mask,
    word_mask_mask,
    return_index=False
):
    '''
        Sparsify/map word based mask onto (word segmentation) token based positions.
    '''
    # 1. Get the sparse token leading subword token locations, and the cumsum of actual words along batch.
    token_sparse_loc = index_to_gather(token_mask*token_mask_mask)
    token_sparse_cumsum = get_cumsum(token_mask*token_mask_mask,
                                     use_mask=True,
                                     prepad=torch.zeros((1), dtype=token_mask.dtype))

    # 2. Get the word mapping locations.
    (batch_index, word_loc) = index_to_gather(word_mask*word_mask_mask,
                                                flat=False, as_tuple=True)
    offsets = token_sparse_cumsum[batch_index]
    offsets = offsets.reshape(word_loc.shape)
    mapping_word_loc_index = word_loc + offsets

    # 3. Finally acquire indices of the sparsified word locations from the mapping locations.
    sparse_word_index = token_sparse_loc[mapping_word_loc_index]

    # 4. Now create sparsified mask.
    sparse_mask = torch.zeros_like(token_mask)
    shape = sparse_mask.shape
    sparse_mask = sparse_mask.reshape(-1) \
                            .index_fill_(dim=0, index=sparse_word_index, value=1) \
                            .reshape(shape)
    if return_index:
        return sparse_mask, sparse_word_index
    else:
        return sparse_mask

def densify_values(data, mask, padding=0):
    '''
        Description: Convert sparse data to dense data.
        Parameters:
            data: Sparse data
            mask: Sparse mask
        Return: data in dense form with padding.
    '''
    # Determine dense row, col size.
    row, col = mask.nonzero(as_tuple=True)
    bc = torch.bincount(row)
    max_bc = torch.max(bc)
    bc_cumsum = torch.cumsum(bc, dim=0)
    bc_cumsum = torch.cat((torch.tensor([0], device=bc_cumsum.device), bc_cumsum), 0)
    # Pad row
    row_exp = torch.arange(start=0, end=bc.shape[0], device=bc.device)[:,None]
    row_exp = row_exp.expand(bc.shape[0], max_bc)
    # print(row_exp)
    # Pad col
    col_exp = [col[start:end] for start, end in zip(bc_cumsum[:-1], bc_cumsum[1:])]
    pads = [torch.zeros(max_bc-(end-start), dtype=torch.int64, device=bc_cumsum.device) \
            for start, end in zip(bc_cumsum[:-1], bc_cumsum[1:])]
    col_exp = torch.stack([torch.cat((c, p)) for c, p in zip(col_exp, pads)])
    # print(col_exp)
    # Gather by row and col
    densed_data = data[row_exp, col_exp]
    densed_data[densed_data<=0] = padding
    return densed_data

def convert_to_weighting_factor(data, mask):
    new_data = data*mask # Zero out padding stuff etc
    max_value = torch.max(new_data, dim=1, keepdim=True)[0]
    distance = ((max_value - new_data) + 1)*mask  # Exclude paddings etc
    distance = distance.float()
    avg = torch.sum(distance, dim=1, keepdim=True) / \
            torch.sum(mask, dim=1, keepdim=True) # Only count on masked bits.
    weight_factor = distance / avg
    return weight_factor

def fill_root_head_loc(
    token_mask,
    token_mask_mask,
    root_head_mask,
    root_head_mask_mask,
    sparse_head,
    fill_value=-100
):
    _, sparse_root_head_index = sparsify_mask(
                                            token_mask,
                                            token_mask_mask,
                                            root_head_mask,
                                            root_head_mask_mask,
                                            return_index=True
                                        )

    # 5. Now remove the root head self-loop edge.
    if sparse_head is not None:
        shape=sparse_head.shape
        sparse_head = sparse_head.reshape(-1) \
                                .index_fill_(dim=0, index=sparse_root_head_index, value=fill_value) \
                                .reshape(shape)

    return (sparse_root_head_index, sparse_head)


def covert_word_to_token_indices(
    token_mask,
    token_mask_mask,
    indices,
    indices_mask,
    indices_mask_mask,
    token_span_count, # subword_span
    token_span_count_mask,
    flat_batch=True,
    indices_offset_n=0,
    batch_offset_required=False,
    ignore_value=-100):
    '''
        Map the word-level indices to the corresponding token-level first token indices of
        words tokenized by the word-segmentation token encoding methods e.g. byte-pair encoding.

        Parameters:
            token_mask:
                The first token mask of each word in the model tokenized sequence.
            token_mask_mask:
                Token batch mask.
            indices:
                out of order word indices and may be a subset of word indices.
                Note that it differs from sparsify_indices.
            indices_mask:
                The first word mask of each entity (by indices).
                An entity is formed of an one or more words.
            indices_mask_mask:
                Indices batch mask.
            token_span_count:
                Token (model tokenized tokens) span of each word.
            token_span_count_mask:
                Token sequence batch mask.
            indices_offset_n:
                Mapping to tokens starts after the model takenizer's prepended tokens (e.g. BOS) 
                assuming that the prepended tokens are single-token-per-word.
                So, right-shift word-level indices by indices_offset_n.
            batch_offset_required:
                Offset indices by batch-wide sequence length if True.
                Has an effect only when flat_batch is True.

        Return:
            A tensor of token indices, and a mask to identify indices' batch dimensions.
    '''
    indices = indices + indices_offset_n

    # Each entry of the token span cumsums corresponds to the first token index (of a word)
    # while the indexing position to the token_span_count_cumsums 'list' itself has one-to-one 
    # mapping to input word-level indices.
    # Thus, acquiring token indices from token_span_count_cumsums can be done
    # simply by using the input word-level indices to slice it.
    # For example, use 2 in the input word-level indices to get the first token index
    # by token_span_count_cumsums[2].
    token_span_count_cumsums = torch.cumsum(token_span_count*token_span_count_mask, dim=-1)
    # Indexing starts from zero.
    pre_pad = torch.zeros((token_span_count_cumsums.shape[0],),
                          dtype=token_span_count_cumsums.dtype,
                          device=token_span_count_cumsums.device)
    token_span_count_cumsums = torch.cat((pre_pad[:,None], token_span_count_cumsums), dim=-1)
    # Ensure not use batch padding values as valid indices.
    valid_size = torch.sum(indices_mask_mask, dim=-1, keepdim=True)

    # Slice indices within the range bound by valid size (i.e. excluding padding at the end).
    nb, _ = token_mask.size()
    token_indices = [token_span_count_cumsums[ib,indices[ib,:valid_size[ib]]] for ib in range(nb)]

    indices_sizes = [indices.shape[0] for indices in token_indices]

    if flat_batch:
        # Create mask to match indices to their batch dimensions, similar to torch.nonzero.
        mask = torch.tensor(list(range(nb)))
        mask = mask.repeat_interleave(torch.tensor(indices_sizes), dim=0)
        if batch_offset_required:
            pre_pad = torch.zeros((1,), dtype=token_span_count_cumsums.dtype, device=token_span_count_cumsums.device)
            batch_offset = get_cumsum(token_mask_mask, use_mask=False, prepad=pre_pad)[:,None]
            token_indices = [token_indices[ib] + batch_offset[ib] for ib in range(nb)]
        token_indices = torch.cat(token_indices, dim=0)
    else:
        max_size = token_mask.shape[-1]
        mask = torch.ones_like(token_mask)
        padded = []
        for ib in range(nb):
            mask[ib, indices_sizes[ib]:] = 0
            pad = torch.ones(max_size-indices_sizes[ib],
                             dtype=token_indices[ib].dtype,
                             device=token_indices[ib].device)*ignore_value
            padded.append(torch.cat((token_indices[ib], pad), dim=-1))
        token_indices = torch.stack(padded, dim=0)

    return token_indices, mask


def get_unique_index(values, indices):
    '''
        Input values should be aleady sorted.
    '''
    _, inv_idx, counts = torch.unique(values, dim=0, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(inv_idx, stable=True)
    cumsum = torch.cat((torch.tensor([0], device=counts.device), counts.cumsum(0)[:-1]))
    # First occurring duplicate location.
    unique_indicies = ind_sorted[cumsum]
    indices = indices.view(-1)[unique_indicies]
    return indices, inv_idx


def get_keys_by_prefix(kwargs, prefix, pop=True):
    keys = [k for k in kwargs.keys() if prefix in k]
    kwargs = {k: kwargs.pop(k) if pop else kwargs.get(k) for k in keys}
    return kwargs
