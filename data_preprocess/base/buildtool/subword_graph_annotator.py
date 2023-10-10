from __future__ import unicode_literals, print_function, division
from buildtool.annotated_utils import cumulative

sanity_check=False

class SubwordGraphAnnotator():
    def __init__(self, ignore_value=-100, depth_padding_id=0, mask_all=False, self_loop=False):
        '''
            mask_all: Also include the word that is not segmented if the flag is True.
        '''
        self.ignore_value = ignore_value
        self.depth_padding_id = depth_padding_id
        self.depth_start_id = self.depth_padding_id + 1 \
            if self.depth_padding_id >= 0 else 0
        self.mask_all = mask_all
        self.self_loop = self_loop

    def __call__(self, subword_mask, output_mask=True, compact=False):
        '''
            output_mask and compact are the mutual exclusive flags.
            e.g.
                subword mask=[1,0,0,0,1,0,0,0,0,1,0]
                range       =[0,1,2,3,4,5,6,7,8,9,10]
                heads       =[0,0,0,0,4,4,4,4,4,9,9]
        '''
        nl = len(subword_mask)
        head = 0
        edges = [self.ignore_value] * nl
        depths = [self.depth_padding_id] * nl
        span_map = {}
        span = [] # Count tokens (subwords) of each word in sequence
        zero_count = 0
        for j in range(nl):
            if subword_mask[j] == 1:
                if zero_count > 0:
                    span[-1] += zero_count
                span += [1]
                # Clear previous word head depth since we don't use it for edge feature embedding.
                head = j
                if self.mask_all and depths[head] == self.depth_padding_id:
                    depths[head] = self.depth_start_id
                    if self.self_loop:
                        edges[head] = head # self loop
                    span_map[head] = 1
                zero_count = 0
            else: # == 0
                if not self.mask_all and depths[head] == self.depth_padding_id:
                    depths[head] = self.depth_start_id
                    if self.self_loop:
                        edges[head] = head # self loop
                    span_map[head] = 1
                edges[j] = head # the following subword points to leading subword.
                depths[j] = depths[j-1] + 1
                span_map[head] += 1
                zero_count += 1

        if sanity_check: # debugging
            import copy
            span_copy = copy.deepcopy(span)
            span_cum = cumulative(span_copy)
            for k, v in span_map.items():
                assert k in span_cum, "span and span map are inconsistent."

        if output_mask:
            edge_mask = list(map(lambda x: int(x != self.ignore_value), edges))
            return {"edge": edges, "depth": depths, "mask": edge_mask,
                    "span": span, "span_map": span_map}
        elif compact:
            edges = list(filter(lambda x: x != self.ignore_value, edges))
            depths = list(filter(lambda x: x != self.depth_padding_id, depths))
        return {"edge": edges, "depth": depths, "span": span, "span_map": span_map}
