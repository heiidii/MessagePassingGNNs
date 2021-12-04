from deeph3.resnets.ResNet2D import ResBlock2D, ResNet2D
import torch.nn as nn
import dgl
from deeph3.graph_models.functions import *
from deeph3.graph_models.layers import Normalize
from deeph3.resnets import ResBlock1D, ResNet1D
import copy
from deeph3.models.AntibodySCAntigenGATComplexAlt.HeteroGraphTransformer \
    import GTransformerHetero
from deeph3.models.AntibodySCAntigenGATComplexAlt.HomoGraphTransformer \
    import GTransformerHomo

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
        
        #print('sg ', sg)
        #for nt in sg.ntypes:
        #    print('sg bnn, bne ', sg.batch_num_nodes(ntype=nt))
        #debug
        #print('gcn bnn,bne ', dict_num_nodes_batch, dict_num_edges_batch)
        #print('sg ', sg, meta_paths)
        #print(sg.edges())
        #src = list(set([t.item() for t in sg.edges()[0]]))
        #print(len(src))
        
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

class InitialEncoding(nn.Module):

    def __init__(self,
                n_hidden,
                n_node_features_ab=20,
                n_edge_features_ab=4,
                n_node_features_ag=20,
                n_edge_features_ag=4,
                n_pe=16,
                num_blocks1D=3,
                num_blocks2D=3,
                kernel_size_ab=17,
                kernel_size_ag=5,
                kernel_size_2d=5,
                use_resnet=False,
                use_2d_resnet=False,
                use_edge_data=True,
                dropout=0.10):

        super(InitialEncoding,self).__init__()

        node_features_ab = n_node_features_ab
        edge_features_ab = n_edge_features_ab
        node_features_ag = n_node_features_ag
        edge_features_ag = n_edge_features_ag
        self.dropout_pe_e = nn.Dropout(dropout)

        if not use_resnet:
                self.dropout_pe = nn.Dropout(dropout)
                self.node_embedding_ab = nn.Sequential(nn.Linear(node_features_ab,n_hidden,bias=True),
                                                        Normalize(n_hidden))
                
                self.node_embedding_ag = nn.Sequential(nn.Linear(node_features_ag,n_hidden,bias=True),
                                                        Normalize(n_hidden))
                
        else:
            self.node_embedding_ab = ResNet1D(node_features_ab,
                                ResBlock1D, [num_blocks1D],
                                init_planes=n_hidden,
                                kernel_size=kernel_size_ab)
            self.node_embedding_ag = ResNet1D(node_features_ag,
                                ResBlock1D, [num_blocks1D],
                                init_planes=n_hidden,
                                kernel_size=kernel_size_ag)
        
        self.use_resnet = use_resnet
        self.use_2d_resnet = use_2d_resnet

        if not use_2d_resnet:
            self.edge_embedding_ab = nn.Sequential(nn.Linear(edge_features_ab,n_hidden,bias=True),
                                                        Normalize(n_hidden))
            self.edge_embedding_ag = nn.Sequential(nn.Linear(edge_features_ag,n_hidden,bias=True),
                                                        Normalize(n_hidden))
        else:
            self.edge_embedding_ab = ResNet2D(edge_features_ab,
                                              ResBlock2D, [num_blocks2D],
                                              init_planes=n_hidden,
                                              kernel_size=kernel_size_2d,
                                              dilation_cycle=1)
            self.edge_embedding_ag = ResNet2D(edge_features_ag,
                                              ResBlock2D, [num_blocks2D],
                                              init_planes=n_hidden,
                                              kernel_size=kernel_size_2d,
                                              dilation_cycle=1)

        self.use_edge_data = use_edge_data

    
    def forward(self,g):

        if 'antib' in g.ntypes:
            nodes_ab, edges_ab = g.nodes['antib'].data['x'], g.edges['ab-ab'].data['y']
            nodes_ag, edges_ag = g.nodes['antigen'].data['x'], g.edges['ag-ag'].data['y']
        else:
            nodes_p0, edges_p0 = g.nodes['p0'].data['x'], g.edges['p0-p0'].data['y']
            nodes_p1, edges_p1 = g.nodes['p1'].data['x'], g.edges['p1-p1'].data['y']
        
        if not self.use_2d_resnet:
            if 'antib' in g.ntypes:
                g.edges['ab-ab'].data['y'] = self.edge_embedding_ab(edges_ab) 
                g.edges['ag-ag'].data['y'] = self.edge_embedding_ag(edges_ag)
            else:
                g.edges['p0-p0'].data['y'] = self.edge_embedding_ag(edges_p0) 
                g.edges['p1-p1'].data['y'] = self.edge_embedding_ag(edges_p1)
        else:
            print('Not implemented yet')
            '''
            needs conversion to adjacency amtrix - 
            however not the same the as in CNN 
            since we do not have a fully
            connected graph
            '''

        if not self.use_resnet:
            if 'antib' in g.ntypes:
                nodes_ab = self.node_embedding_ab(nodes_ab)
                nodes_ag = self.node_embedding_ag(nodes_ag)
                g.nodes['antib'].data['x'] = nodes_ab + \
                                                        self.dropout_pe(g.nodes['antib'].data['pe'])
                g.nodes['antigen'].data['x'] = nodes_ag + \
                                                        self.dropout_pe(g.nodes['antigen'].data['pe'])
            else:
                #shared encodings
                nodes_p0 = self.node_embedding_ag(nodes_p0)
                nodes_p1 = self.node_embedding_ag(nodes_p1)
                g.nodes['p0'].data['x'] = nodes_p0 + \
                                                        self.dropout_pe(g.nodes['p0'].data['pe'])
                g.nodes['p1'].data['x'] = nodes_p1 + \
                                                        self.dropout_pe(g.nodes['p1'].data['pe'])
        else:
            if 'antib' in g.ntypes:
                nodes_ab.unsqueeze_(0)
                nodes_ag.unsqueeze_(0)
                nodes_ab =  self.node_embedding_ab(nodes_ab.permute(0,2,1))
                nodes_ag =  self.node_embedding_ag(nodes_ag.permute(0,2,1))
                nodes_ab = nodes_ab.squeeze_(0).permute(1,0)
                nodes_ag = nodes_ag.squeeze_(0).permute(1,0)
                g.nodes['antib'].data['x'] = nodes_ab 
                g.nodes['antigen'].data['x'] = nodes_ag
            else:
                nodes_p0.unsqueeze_(0)
                nodes_p1.unsqueeze_(0)
                nodes_p0 =  self.node_embedding_ag(nodes_p0.permute(0,2,1))
                nodes_p1 =  self.node_embedding_ag(nodes_p1.permute(0,2,1))
                nodes_p0 = nodes_p0.squeeze_(0).permute(1,0)
                nodes_p1 = nodes_p1.squeeze_(0).permute(1,0)
                g.nodes['p0'].data['x'] = nodes_p0
                g.nodes['p1'].data['x'] = nodes_p1

        return g

