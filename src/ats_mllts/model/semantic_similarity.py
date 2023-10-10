from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.biaffine import BiAffineTransform

sanity_check=False

class SemanticSimilarityMeasure(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.lambda_w = options["lambda"]
        rel_size = options["tgt_mlp_dims"][-1]
        self.rel_attn = BiAffineTransform(
                            rel_size, rel_size,
                            n_label=1,
                            h_bias=False, s_bias=False,
                            init_value=1.)

    def forward(
        self,
        h_x,
        h_y,
        x_mask,
        y_mask,
        reduction=True
    ):
        # Vector similarity per prediction
        xy_struct_loss = self.compute_flow_cost(h_x,
                                                x_mask[:,:h_x.shape[1]],
                                                h_y,
                                                y_mask[:,:h_y.shape[1]],
                                                reduction)
        return xy_struct_loss * self.lambda_w

    def compute_flow_cost(self, src, src_mask, pred, pred_mask, reduction):
        cost_mx = src.unsqueeze(dim=-3) - pred.unsqueeze(dim=-2)
        cost_mx = torch.norm(cost_mx, p=2, dim=-1, keepdim=True).squeeze(-1)
        cost_mask = src_mask.unsqueeze(dim=-2) * pred_mask.unsqueeze(dim=-1)
        tr_mx = self.rel_attn(pred, src)
        tr_mx = tr_mx.permute(2,0,1)
        tr_mx.masked_fill_(cost_mask==0, -1e10)
        tr_prob = F.softmax(tr_mx, dim=-1)
        flow_cost = tr_prob*cost_mx
        if reduction:
            nL1 = torch.sum(pred_mask, dim=-1, keepdim=True)
            nL2 = torch.sum(src_mask, dim=-1, keepdim=True)
            if sanity_check:
                flow_cost_test = torch.sum(flow_cost, dim=2, keepdim=False)
                flow_cost_test = flow_cost_test / nL2
                flow_cost_test = torch.sum(flow_cost_test, dim=1, keepdim=False)
                flow_cost_test = flow_cost_test / nL1
                flow_cost_test = torch.mean(flow_cost_test)
            flow_cost = torch.sum(flow_cost, dim=(1,2), keepdim=True) / (nL1*nL2) # average over batch and src
            flow_cost = torch.mean(flow_cost)
        else:
            # _, nL1, nL2 = cost_mx.shape
            flow_cost = torch.sum(flow_cost, dim=(1,2), keepdim=True) #/ (nL1*nL2) # average over batch and src
            flow_cost = flow_cost.squeeze(dim=-1)
        return flow_cost
