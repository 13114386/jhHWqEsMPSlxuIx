from __future__ import unicode_literals, print_function, division
import itertools
import numpy as np
from buildtool.annotated_utils import cumulative

def forward_match(values, query, cum_spans):
    query_locs = []
    for q, (start, end) in zip(query, zip(cum_spans[:-1], cum_spans[1:])):
        loc = values[start:end].index(q)
        query_locs.append(loc+start)
    return query_locs


def merge_coref_by_antecedent(corefs, sentence_sizes, logger=None):
    '''
        coref = 
        {
            "sent_num": current, # one based rather than zero based.
            "coref_index": coref_index,
            "repr_mask": repr_mask,
            "entity_mask": entity_mask,
            "entity_span": entity_span,
            etc...
        }
    '''
    cumsum = cumulative(sentence_sizes)
    data = []
    for coref in corefs: # A list of antecedents and their mentions
        # Offset indices w.r.t. the article.
        sent_nums = [[sent_num]*n for n, sent_num in zip(coref["entity_span"], coref["sent_num"])]
        sent_nums = list(itertools.chain(*sent_nums))
        offsets = [cumsum[sent_num-1] for sent_num in sent_nums] # sent_num is one-based.
        # Get head locations in coref index list.
        cum_spans = cumulative(coref["entity_span"])
        head_loc = forward_match(coref["coref_index"], coref["coref_head_index"], cum_spans)
        # Offset all coref indices
        offset_coref_index = [ci+offset for ci, offset in zip(coref["coref_index"], offsets)]
        # Gather offsetted head indices from offsetted coref indices by head locations.
        np_offset_index = np.array(offset_coref_index)
        offset_head_index = list(np_offset_index[head_loc])
        # Convert np.int32 to int for json.dump.
        offset_head_index = [int(h) for h in offset_head_index]
        if len(coref["entity_span"]) != len(offset_head_index):
            raise ValueError(f"Failed to reconcile indices between entity span and offsetted head index.")

        data.append({
            "coref_head_index": offset_head_index,
            "coref_index": offset_coref_index,
            "coref_repr_mask": coref["repr_mask"],
            "coref_entity_mask": coref["entity_mask"],
            "coref_entity_span": coref["entity_span"],
            "coref_sent_num": coref["sent_num"],
            "coref_type": coref["type"],
            "coref_number": coref["number"],
            "coref_animacy": coref["animacy"],
            "coref_gender": coref["gender"],
        })
    return data


def merge_all_corefs(corefs, keep_dict=True):
    '''
        Merge all antecedents
        For antecedents of an article, their coreference indices mask
        and their sentence nums mask can be used to distinguish
        each antecedent context.
        Each antecedent-coreferents boundary can be identified by
        coref_repr_mask*coref_entity_mask and coref_sent_num_mask.
        coref_repr_mask identifies which indices are antecedent entities.
        coref_entity_mask masks the leading location of multi-word entity indices.
        coref_sent_num_mask masks which sentence numbers associate with the antecedent-coreferents boundary.
    '''
    # Create coref_sent_num_mask to distinguish sentence spans of
    # different antecedents and their mentions.
    new_key_map = {"coref_sent_num": "coref_sent_num_mask"}
    new_mask_columns = {v: [] for v in new_key_map.values()}
    column_keys = list(zip(*[coref.keys() for coref in corefs]))
    column_keys = [key[0] for key in column_keys] + list(new_key_map.values())
    column_view = {k: [] for k in column_keys}
    column_view = {**column_view, **new_mask_columns}
    for coref in corefs:
        for k, v in coref.items():
            column_view[k] += v
            if k in new_key_map:
                column_view[new_key_map[k]] += [1] + [0]*(len(v)-1)

    if not keep_dict:
        column_view = list(column_view.values())

    return column_view
