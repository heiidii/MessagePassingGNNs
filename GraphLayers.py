
import torch.nn as nn
import dgl
from graph_models.functions import *
from graph_models.layers import Normalize
import copy
from HeteroGraphTransformer import GTransformerHetero
from HomoGraphTransformer import GTransformerHomo

class HeteroGCNConv(nn.Module):
    def __init__(self, in_size, out_size, n_layers=3, dropout=0.1):

        super(HeteroGCNConv, self).__init__()

        self.gcn_layers = nn.ModuleList()
        self.norm_edge_weight = dgl.nn.pytorch.conv.EdgeWeightNorm(norm='both')
        for i in range(n_layers):
            self.gcn_layers.append( \
                        dgl.nn.pytorch.conv.GraphConv(in_size,
                                                        out_size,
                                                        norm='none',
                                                        activation=nn.functional.elu))
        
    def forward(self, g, meta_paths, ntype, key, edge_weight=None):
        
        dict_num_nodes_batch = {}
        for nt in g.ntypes:
            dict_num_nodes_batch[nt] = g.batch_num_nodes(ntype=nt)

        dict_num_edges_batch = {}
        for et in meta_paths:
            dict_num_edges_batch[et] = g.batch_num_edges(etype=et)

        sg = dgl.metapath_reachable_graph(g, meta_paths)
        sg.set_batch_num_nodes(dict_num_nodes_batch)
        sg.set_batch_num_edges(dict_num_edges_batch)
        
        norm_weight = self.norm_edge_weight(sg, edge_weight.squeeze(1))
        feats = sg.nodes[ntype].data[key]
        for layer in self.gcn_layers:
            feats = layer(sg, feats, edge_weight=norm_weight)
        
        return feats


class HeteroGATConv(nn.Module):
    '''
    Internally, if a bipartite graph is supplied, 
    GATConv applies two different weight matrices (desired)
    Here is the source code:
    if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
    else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
    copied from: \
        https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    So if needed, we can use this module for ab-ag type if attention as well    
    '''
    def __init__(self, in_size, out_size, layer_num_heads=4, \
                                n_layers = 3, dropout = 0.15,\
                                residual = True):

        super(HeteroGATConv, self).__init__()
        #self.norm = Normalize(out_size)
        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gat_layers.append( \
                            dgl.nn.pytorch.conv.GATConv(in_size,
                                                        int(out_size/layer_num_heads),
                                                        num_heads=layer_num_heads,
                                                        attn_drop=dropout,
                                                        #feat_drop=dropout,
                                                        activation=nn.functional.elu,
                                                        residual=residual,
                                                        allow_zero_in_degree=True))
        
    def forward(self, g, meta_paths, ntype, key, get_attention=True):

        if not meta_paths == []:

            dict_num_nodes_batch = {}
            for nt in g.ntypes:
                dict_num_nodes_batch[nt] = g.batch_num_nodes(ntype=nt)

            dict_num_edges_batch = {}
            for et in meta_paths:
                dict_num_edges_batch[et] = g.batch_num_edges(etype=et)
            

            sg = dgl.metapath_reachable_graph(g, meta_paths)
            sg.set_batch_num_nodes(dict_num_nodes_batch)
            sg.set_batch_num_edges(dict_num_edges_batch)
            #debug
            #print('gcn bnn,bne ', dict_num_nodes_batch, dict_num_edges_batch)
            #print('sg ', sg, meta_paths)
            #print(sg.edges())
            #src = list(set([t.item() for t in sg.edges()[0]]))
            #print(len(src))

        else:
            sg = g
        
        if type(ntype) is tuple:
            feats = (sg.nodes[ntype[0]].data[key[0]],
                     sg.nodes[ntype[1]].data[key[1]])

            for layer in self.gat_layers[:-1]:
                feats_dst = layer(sg,feats).flatten(1)
                feats = tuple([feats[0],feats_dst])
        else:
            if not ntype == '':
                feats = sg.nodes[ntype].data[key]
            else:
                feats = sg.ndata[key]
            
            for layer in self.gat_layers[:-1]:
                feats = layer(sg, feats).flatten(1)

        feats, attention = self.gat_layers[-1](sg, feats, get_attention=get_attention)
        
        return feats.flatten(1), attention

class HomoGAT(nn.Module):
    def __init__(self, n_hidden, n_layers, n_heads, dropout):
        super(HomoGAT, self).__init__()

        self.gat_block = HeteroGATConv(n_hidden,n_hidden,n_heads,n_layers,dropout)

    def forward(self, g):

        homo_g = dgl.to_homogeneous(g,ndata=['x'])
        homo_g.ndata['x'], _ = self.gat_block(homo_g, [], '', 'x')
        hetero_g = dgl.to_heterogeneous(homo_g, g.ntypes, g.etypes)

        return [hetero_g.nodes[ntype].data['x'] for ntype in g.ntypes]


class MultiLayerHeteroGT(nn.Module):
    def __init__(self, n_hidden, n_heads, n_layers, dropout=0.15):
        super().__init__()

        gt_layer = GTransformerHetero(n_hidden, n_heads, dropout=dropout)
        self.gt_block = nn.ModuleList()
        for _ in range(n_layers):
            self.gt_block.append(copy.deepcopy(gt_layer))

    def forward(self, g, ntype, etype, use_topk=False):

        for gt_layer in self.gt_block:
        
            feats = gt_layer(g, ntype, etype, use_topk=use_topk)
            g.nodes[ntype].data['x'] = feats
        
        return feats


class MultiLayerHomoGT(nn.Module):
    def __init__(self, n_hidden, n_heads, n_layers, dropout=0.15):
        super().__init__()

        gt_layer = GTransformerHomo(n_hidden, n_heads, dropout=dropout)
        self.gt_block = nn.ModuleList()
        for _ in range(n_layers):
            self.gt_block.append(copy.deepcopy(gt_layer))

    def forward(self, g, debug=False, use_topk=False):

        for gt_layer in self.gt_block:
        
            if debug:
                feats_list, attn = gt_layer(g, debug=debug, use_topk=use_topk)
            else:
                feats_list = gt_layer(g, use_topk=use_topk)
            for ntype, feats in zip(g.ntypes, feats_list):
                g.nodes[ntype].data['x'] = feats
        
        if debug:
            return feats_list, attn
        else:
            return feats_list
