from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mlp import MLP
from torch_geometric.nn import GeneralConv
from model.gnn_datatransform import GNNDataTransform, DependencyIndexer
from model.pna_gnn import PNANet
from model.datautil import (convert_to_weighting_factor,
                            flat_gather)
from model.modelutil import create_embedding

class FeatureRepr(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.mlp = MLP(opts["layer_ndims"],
                        activation=eval("F."+opts["activation"]),
                        batch_norm=opts["batch_norm"],
                        dropout=opts["dropout"])

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.mlp(x)
        return x


class Factum(nn.Module):
    def __init__(
        self,
        opt,
        struct_opt,
        feat_vocab=None,
        inarc_embedding_shared=None,
        logger=None
    ):
        super().__init__()
        in_edge_channels = None
        if opt["edge_feature"]:
            assert inarc_embedding_shared is not None or feat_vocab is not None, \
                    "Factum can not have both shared inarc embedding and inarc vocab none"
            if inarc_embedding_shared:
                self.inarc_embedding = inarc_embedding_shared
            else:
                pad_id = struct_opt["inarc_map"]['pad']
                dim = struct_opt["inarc_map"]['dim']
                self.inarc_embedding = create_embedding(
                    "inarc", feat_vocab, pad_id, dim, logger)

            in_edge_channels = struct_opt["inarc_map"]['dim']

        self.src_MLP = MLP(opt["src_mlp_dims"], F.leaky_relu, opt["mlp_drop"], opt["mlp_bias"])
        self.tgt_MLP = MLP(opt["tgt_mlp_dims"], F.leaky_relu, opt["mlp_drop"], opt["mlp_bias"])
        rel_size = opt["tgt_mlp_dims"][-1]
        if opt["model"] == "general":
            self.sim_gnn = GeneralConv(in_channels=rel_size,
                                        out_channels=rel_size,
                                        in_edge_channels=in_edge_channels,
                                        skip_linear=True,
                                        directed_msg=True,
                                        attention=opt["attention"],
                                        l2_normalize=True)
        else:
            self.sim_gnn = PNANet(in_channels=rel_size,
                                    out_channels=rel_size,
                                    edge_dim=in_edge_channels,
                                    n_layers=opt["n_layers"],
                                    scalers=opt["scalers"],
                                    directed_msg=opt["directional"])
        self.transform = GNNDataTransform()
        self.dep_indexer = DependencyIndexer()
        if opt["graph_emb"]["choice"] == "axis_feature":
            axis_opt = opt["graph_emb"]["axis_feature"]
            self.feature_repr = FeatureRepr(axis_opt)

        self.opt = opt

    def forward(self, inputs, training=True):
        '''
            Expect batch first format
        '''
        # H -> hidden size, L -> sentence length, B -> batch size
        # Source doc structure
        h_x = inputs["h_x"]
        x_mask = inputs["x_mask"] # attention mask
        src_token_dense_mask = inputs["src_token_dense_mask"] # attention mask has special tokens masked out
        src_token_sparse_mask = inputs["src_token_sparse_mask"] # token_mask
        src_token_inarc_type = inputs["src_token_inarc_type"]
        src_token_arc_head = inputs["src_token_arc_head"]
        src_token_depth = inputs["src_token_depth"]
        src_word_inarc_type = inputs["src_word_inarc_type"] # word level dense form
        src_word_inarc_type_mask = inputs["src_word_inarc_type_mask"]
        # Match length with structure data
        dim = 1 # sentence dim
        n_excl = h_x.shape[dim] - src_token_inarc_type.shape[dim]
        assert n_excl == 0 or n_excl == 2
        h_x = h_x[:,n_excl//2:-n_excl//2] if n_excl == 2 else h_x
        x_mask = x_mask[:,n_excl//2:-n_excl//2] if n_excl == 2 else x_mask
        # Convert depth for node weights such that the less deep, the more important
        src_token_weight = convert_to_weighting_factor(src_token_depth,
                                                 src_token_sparse_mask*src_token_dense_mask)

        X_g = self.src_MLP(h_x, training=training)
        src_repr = self.compute_graph_repr(self.transform,
                                            self.dep_indexer,
                                            data=X_g,
                                            attention_mask=x_mask,
                                            token_arc_head=src_token_arc_head,
                                            token_arc_type=src_token_inarc_type,
                                            word_arc_type=src_word_inarc_type,
                                            word_arc_type_mask=src_word_inarc_type_mask,
                                            token_dense_mask=src_token_dense_mask,
                                            token_sparse_mask=src_token_sparse_mask,
                                            token_node_weight=src_token_weight,
                                            training=training)

        if self.opt["graph_emb"]["choice"] == "axis_feature":
            src_repr = self.feature_repr(src_repr)

        # Reference ground truth structure
        h_y = inputs["h_y"]
        y_mask = inputs["y_mask"]
        tgt_token_dense_mask = inputs["tgt_token_dense_mask"] # Mask out padding stuff
        tgt_token_sparse_mask = inputs["tgt_token_sparse_mask"] # Also mask out the others of not interest
        tgt_token_inarc_type = inputs["tgt_token_inarc_type"]
        tgt_token_arc_head = inputs["tgt_token_arc_head"]
        tgt_token_depth = inputs["tgt_token_depth"]
        tgt_word_inarc_type = inputs["tgt_word_inarc_type"]
        tgt_word_inarc_type_mask = inputs["tgt_word_inarc_type_mask"]
        # Match length with structure data
        dim = 1 # sentence dim
        n_excl = h_y.shape[dim] - tgt_token_inarc_type.shape[dim]
        assert n_excl == 0 or n_excl == 2
        h_y = h_y[:,:-n_excl] if n_excl == 2 else h_y
        y_mask = y_mask[:,n_excl//2:-n_excl//2] if n_excl == 2 else y_mask
        # Convert depth for node weights such that the less deep, the more important
        tgt_token_weight = convert_to_weighting_factor(tgt_token_depth,
                                                 tgt_token_sparse_mask*tgt_token_dense_mask)

        X_g = self.tgt_MLP(h_y, training=training)
        y_repr = self.compute_graph_repr(self.transform,
                                         self.dep_indexer,
                                         data=X_g,
                                         attention_mask=y_mask,
                                         token_arc_head=tgt_token_arc_head,
                                         token_arc_type=tgt_token_inarc_type,
                                         word_arc_type=tgt_word_inarc_type,
                                         word_arc_type_mask=tgt_word_inarc_type_mask,
                                         token_dense_mask=tgt_token_dense_mask,
                                         token_sparse_mask=tgt_token_sparse_mask,
                                         token_node_weight=tgt_token_weight,
                                         training=training)

        if self.opt["graph_emb"]["choice"] == "axis_feature":
            y_repr = self.feature_repr(y_repr)

        return (src_repr, y_repr)

    def compute_graph_repr(self, transform, dep_indexer,
                            data, attention_mask, 
                            token_arc_head, token_arc_type,
                            word_arc_type, word_arc_type_mask,
                            token_dense_mask, token_sparse_mask,
                            token_node_weight, training):
        '''
            Compute graph representation of sequence dependency structure.

            transform:              Data transformation object.
            dep_indexer:            Dependency data indexing object.
            data:                   Latent data.
            token_arc_head:         Sparsified dependency head of syntactic arc.
            token_arc_type:         Sparsified dependency type of syntactic arc.
            attention_mask:         Inlucde special tokens such as BOS and EOS.
                                    It is created by a tokenizer.
            token_dense_mask:       Mask out the special tokens from attention_mask.
            token_sparse_mask:      Sparse mask that mask the first token of word segmentation.
                                    It may include the special tokens.
            token_node_weight:      Weighting factor of nodes.
        '''
        sparse_mask = token_sparse_mask*token_dense_mask
        results = dep_indexer(mask=attention_mask,
                            sparse_mask=sparse_mask)
        indexer = results["indexer"]
        cumsum = results["cumsum"]
        edges = transform.make_edges(token_arc_head,
                                     indexer, cumsum, order=self.opt["order"])
        edge_features = None
        if self.opt["edge_feature"]:
            edge_features = self.inarc_embedding(word_arc_type*word_arc_type_mask)
            # Gather edge features.
            edge_features = flat_gather(edge_features, word_arc_type_mask)
        data_g = transform.debatch(data, indexer=None)
        node_weight_g = transform.debatch(token_node_weight[...,None], indexer=None)
        sim_model_spec = self.opt["model"]
        if sim_model_spec == "general":
            repr = self.sim_gnn.forward(x=data_g,
                                        edge_index=edges,
                                        edge_feature=edge_features)
        elif sim_model_spec == "PNANet":
            repr = self.sim_gnn.forward(x=data_g,
                                        edge_index=edges,
                                        edge_attr=edge_features,
                                        node_weight=node_weight_g[...,None])
        else:
            msg = "The chosen similarity model spec ({}) is not supported.".format(sim_model_spec)
            raise NotImplementedError(msg)
        repr = repr.reshape(data.shape)
        return repr
