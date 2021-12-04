import copy
import torch
import torch.nn as nn
from torch.nn.functional import dropout
from deeph3.graph_models.functions import *
from deeph3.models.AntibodySCAntigenGATComplexAlt.EdgeLayers\
                    import EdgeNetHomoLayer, EdgeNetLayer, EdgeNetSimpleLayer,\
                    EdgeNetSimpleHomoLayer, EdgeNetfromEdgeHomoLayer
from deeph3.models.AntibodySCAntigenGATComplexAlt.HomoGraphTransformer \
                    import GTransformerHomo
from deeph3.models.AntibodySCAntigenGATComplexAlt.GraphLayers\
                    import HeteroGATConv, HomoGAT, MultiLayerHeteroGT, MultiLayerHomoGT
from deeph3.models.AntibodySCAntigenGATComplexAlt.HeteroGraphTransformer \
    import GTransformerHetero

class SuSNEGBlock(nn.Module):
    """
    """
    def __init__(self,
                 n_hidden,
                 n_heads,
                 n_layers,
                 dropout=0.15,
                 blocks=3,
                 num_dist_bins_int=10,
                 last_all=False,
                 add_pe=True,
                 shared_cross_attn=True,
                 cross_path_only=False,
                 combined_attn=False,
                 gat_inside_block=True,
                 classify_nodes=False,
                 dropout_last=0.20,
                 multilayer_gt=False
                ):
        
        super(SuSNEGBlock, self).__init__()

        #Decoder
        self.n_hidden = n_hidden
        self.blocks = blocks
        self.last_all = last_all
        self.add_pe = add_pe
        self.shared_cross_attn = shared_cross_attn
        self.cross_path_only = cross_path_only
        self.combined_attn = combined_attn
        self.gat_inside_block = gat_inside_block
        self.classify_nodes = classify_nodes

        if self.gat_inside_block:
            if shared_cross_attn: #default
                self.decoder_abag = nn.ModuleList([HeteroGATConv((n_hidden,n_hidden),
                                                                  n_hidden,n_heads,n_layers,dropout) \
                                                                    for _ in range(self.blocks)])
            elif combined_attn:
                self.shared_decoder_abag = nn.ModuleList([HomoGAT(n_hidden,n_layers,n_heads,dropout)
                                                             for _ in range(self.blocks)])
            else:
                self.decoder_abag = nn.ModuleList([HeteroGATConv((n_hidden,n_hidden),n_hidden,n_heads,n_layers,dropout) \
                                                                                for _ in range(self.blocks)])
                self.decoder_agab = nn.ModuleList([HeteroGATConv((n_hidden,n_hidden),n_hidden,n_heads,n_layers,dropout) \
                                                                                for _ in range(self.blocks)])
        else:
            if shared_cross_attn:
                self.decoder_abag = HeteroGATConv((n_hidden, n_hidden),
                                                  n_hidden, n_heads,
                                                  n_layers, dropout)
            elif combined_attn:
                self.shared_decoder_abag = HomoGAT(n_hidden, n_layers, n_heads, dropout)
            else:
                self.decoder_abag = HeteroGATConv((n_hidden, n_hidden),
                                                  n_hidden, n_heads,
                                                  n_layers, dropout)
                self.decoder_agab = HeteroGATConv((n_hidden, n_hidden),
                                                  n_hidden, n_heads,
                                                  n_layers, dropout)

        self.update_int_edges = nn.ModuleList([EdgeNetSimpleLayer(n_hidden,
                                                                  n_hidden)
                                               for _ in range(self.blocks)])

        if not multilayer_gt:
            self.update_graph = nn.ModuleList([GTransformerHomo(n_hidden, n_heads, dropout=dropout) 
                                           for _ in range(self.blocks)])
        else:
            self.update_graph = nn.ModuleList([MultiLayerHomoGT(n_hidden, n_heads, 2,
                                                                dropout=dropout)
                                                for _ in range(self.blocks)])

        self.update_all_edges_final = nn.ModuleList([
                                        EdgeNetSimpleHomoLayer(n_hidden,
                                                               n_hidden,
                                                               dropout_last)
                                        for _ in range(self.blocks)])


    def forward(self, g, out_etypes=['ab-ag', 'ag-ab'],
                in_etypes=['ab-ab', 'ag-ag'], debug=False):

        # 1. Cross Attention Nodes GAT
        if not self.cross_path_only:
            meta_path_1 = [out_etypes[0], in_etypes[1]]
            meta_path_0 = [out_etypes[1], in_etypes[0]]
        else:
            meta_path_1 = [out_etypes[0]]
            meta_path_0 = [out_etypes[1]]

        if not self.gat_inside_block:
            
            if self.add_pe:
                for ntype in g.ntypes:
                    g.nodes[ntype].data['x'] =  g.nodes[ntype].data['x'] + \
                                                g.nodes[ntype].data['pe']
            if self.shared_cross_attn:
                ntype_0, ntype_1 = g.ntypes[0], g.ntypes[1]
                g.nodes[ntype_0].data['x'], attention =\
                    self.decoder_abag(g, meta_path_0, (ntype_1, ntype_0), ('x', 'x'))
                g.nodes[ntype_1].data['x'], attention =\
                    self.decoder_abag(g, meta_path_1, (ntype_0, ntype_1), ('x', 'x'))

            elif self.combined_attn:
                feats_list = self.shared_decoder_abag(g)
                for ntype, feats in zip(g.ntypes, feats_list):
                    g.nodes[ntype].data['x'] = feats
            
            else:
                ntype_0, ntype_1 = g.ntypes[0], g.ntypes[1]
                g.nodes[ntype_0].data['x'], attention = self.decoder_agab(g, meta_path_0,
                                                        (ntype_1, ntype_0),('x','x'))
                
                g.nodes[ntype_1].data['x'], attention = self.decoder_abag[j](g, meta_path_1,\
                                                            (ntype_0,ntype_1),('x','x'))
        edges_int = []
        edges_all = []
        for j in range(self.blocks):

            # 0.
            
            if self.gat_inside_block:
                if self.add_pe:
                    for ntype in g.ntypes:
                        g.nodes[ntype].data['x'] =  g.nodes[ntype].data['x'] + \
                                                    g.nodes[ntype].data['pe']

                if self.shared_cross_attn:
                    ntype_0, ntype_1 = g.ntypes[0], g.ntypes[1]
                    g.nodes[ntype_0].data['x'], attention = self.shared_decoder_abag[j](g,meta_path_0,\
                                                                (ntype_1, ntype_0),('x','x'))
                    g.nodes[ntype_1].data['x'], attention = self.shared_decoder_abag[j](g, meta_path_1,\
                                                                (ntype_0,ntype_1),('x','x'))
                elif self.combined_attn:
                        feats_list = self.shared_decoder_abag[j](g)
                        for ntype,feats in zip(g.ntypes, feats_list):
                            g.nodes[ntype].data['x'] = feats
                else:
                    g.nodes[ntype_0].data['x'], attention = self.decoder_agab[j](g, meta_path_0,
                                                            (ntype_1, ntype_0),('x','x'))
                    
                    g.nodes[ntype_1].data['x'], attention = self.decoder_abag[j](g, meta_path_1,\
                                                                (ntype_0,ntype_1),('x','x'))
                
            # 2. Residual update loop-epi edges from nodes
            efeats_list = self.update_int_edges[j](g,out_etypes,nkey='x',residual=j>0)
            for et,ef in zip(out_etypes,efeats_list):
                g.edges[et].data['y'] = ef
            
            if debug:
                edges_int.append(efeats_list)

            # 3. Update nodes from nodes+edges data - GTransformer UniMP paper
            # TODO: maybe 3. alternative: Use MPNN from Ingraham Jaakkola work
            if debug:
                feats_list, attn_ug = self.update_graph[j](g, debug=debug)
            else:
                feats_list = self.update_graph[j](g)
            for ntype,feats in zip(g.ntypes, feats_list):
                g.nodes[ntype].data['x'] = feats

            if self.classify_nodes and j==(self.blocks-1):
                #skip last edge update block if predicting nodes
                continue

            # 4. update all edges from new node reps
            #try gated residual
            efeats_list = self.update_all_edges_final[j](g)
            for (s,et,d),efeat in zip(g.canonical_etypes,efeats_list):
                g.edges[et].data['y'] = efeat
            
            if debug:
                edges_all.append(efeats_list)
        
        if debug:
            return g, edges_int, edges_all, attn_ug
        return g


class SNEGBlock(nn.Module):
    """
    """
    def __init__(self,
                n_hidden,
                n_heads,
                n_layers,
                dropout=0.15,
                blocks=3,
                num_dist_bins_int=10,
                last_all=False,
                add_pe=True,
                shared_cross_attn=True, #moved to forward
                cross_path_only=False,
                combined_attn=False,
                dropout_last=2.0,
                multilayer_gt=False,
                update_edge_from_edge=False,
                update_nodes_with_cross_attn=True,
                apply_attn_to_edges=False,
                update_edges_in_gt=False,
                cross_attn_gt=False,
                use_topk_sorting=False
                ):
        super(SNEGBlock,self).__init__()

        #Decoder
        self.n_hidden = n_hidden
        self.blocks = blocks
        self.last_all = last_all
        self.add_pe = add_pe
        self.shared_cross_attn = shared_cross_attn
        self.cross_path_only = cross_path_only
        self.combined_attn = combined_attn
        self.update_edge_from_edge = update_edge_from_edge
        self.update_nodes_with_cross_attn = update_nodes_with_cross_attn
        self.apply_attn_to_edges = apply_attn_to_edges
        #NOTE: needs to be tested - add arg to common_utils for testing
        self.update_edges_in_gt = update_edges_in_gt
        self.update_nodes_with_gt_cross_attn = cross_attn_gt
        self.use_topk = use_topk_sorting

        if self.update_nodes_with_cross_attn:
            if combined_attn:
                self.decoder_abag = nn.ModuleList([HomoGAT(n_hidden,n_layers,n_heads,dropout) for _ in range(self.blocks)])
            else:
                self.decoder_abag = nn.ModuleList([HeteroGATConv((n_hidden,n_hidden),n_hidden,n_heads,n_layers,dropout) \
                                                                                for _ in range(self.blocks)])
                self.decoder_agab = nn.ModuleList([HeteroGATConv((n_hidden,n_hidden),n_hidden,n_heads,n_layers,dropout) \
                                                                                for _ in range(self.blocks)])
        if self.update_nodes_with_gt_cross_attn:
            self.decoder_abag = nn.ModuleList(
                                [MultiLayerHeteroGT(n_hidden, n_heads, 2, dropout) \
                                    for _ in range(self.blocks)] )
            self.decoder_agab = nn.ModuleList(
                                [MultiLayerHeteroGT(n_hidden, n_heads, 2, dropout) \
                                    for _ in range(self.blocks)] )

        self.update_int_edges = nn.ModuleList([EdgeNetLayer(n_hidden, n_hidden, dropout) for _ in range(self.blocks)])

        self.update_all_edges = nn.ModuleList([EdgeNetHomoLayer(n_hidden, n_hidden, dropout) for _ in range(self.blocks)])
        if update_edge_from_edge:
            self.update_all_edges_from_edges = nn.ModuleList([EdgeNetfromEdgeHomoLayer(n_hidden, n_hidden, dropout) for _ in range(self.blocks)])

        if not multilayer_gt:
            self.update_graph = nn.ModuleList([GTransformerHomo(n_hidden, n_heads, dropout, update_edges=self.update_edges_in_gt)
                                                for _ in range(self.blocks)])
        else:
            self.update_graph = nn.ModuleList([MultiLayerHomoGT(n_hidden, n_heads, 2,
                                                                dropout=dropout)
                                                for _ in range(self.blocks)])

        if self.last_all:
            self.update_all_edges_final = nn.ModuleList([
                                            EdgeNetHomoLayer(n_hidden, n_hidden, dropout_last)
                                                for _ in range(self.blocks)])
            if update_edge_from_edge:
                self.update_all_edges_from_edges_final = nn.ModuleList([EdgeNetfromEdgeHomoLayer(n_hidden, n_hidden, dropout)
                                                                         for _ in range(self.blocks)])
        else:
            self.update_int_edges_final = nn.ModuleList([
                                            EdgeNetLayer(n_hidden, n_hidden, dropout_last)
                                                for _ in range(self.blocks)])
        


    def forward(self,g, out_etypes = ['ab-ag', 'ag-ab'],
                in_etypes = ['ab-ab', 'ag-ag'], debug=False,
                shared_cross_attn=True):
        edges_int = []
        edges_all = []
        ntype_0, ntype_1 = g.ntypes[0], g.ntypes[1]

        cross_edges = g.edges(etype=out_etypes[0],form='all', order='eid')
        pred_node_1 = list(set([t.item() for t in cross_edges[0]]))
        pred_node_1.sort()
        pred_node_0 = list(set([t.item() for t in cross_edges[1]]))
        pred_node_0.sort()

        for j in range(self.blocks):

            if self.add_pe:
                for ntype in g.ntypes:
                    g.nodes[ntype].data['x'] =  g.nodes[ntype].data['x'] + \
                                                 g.nodes[ntype].data['pe']

            if self.update_nodes_with_cross_attn:
                # 1. Cross Attention Nodes GAT
                if not self.cross_path_only:
                    meta_path_1 = [out_etypes[0], in_etypes[1]]
                    meta_path_0 = [out_etypes[1], in_etypes[0]]
                else:
                    meta_path_1 = [out_etypes[0]]
                    meta_path_0 = [out_etypes[1]]

                #runtime: shared_cross_attn - \
                # can be used to distinguish b/w p-p and ab-ag
                if shared_cross_attn:
                    #reusing/sharing layer parameters
                    g.nodes[ntype_0].data['x'], attention_1 = self.decoder_abag[j](g,meta_path_0,\
                                                                (ntype_1, ntype_0),('x','x'))
                    g.nodes[ntype_1].data['x'], attention_0 = self.decoder_abag[j](g, meta_path_1,\
                                                                (ntype_0,ntype_1),('x','x'))
                else:
                    g.nodes[ntype_0].data['x'], attention_1 = self.decoder_agab[j](g, meta_path_0,
                                                            (ntype_1, ntype_0),('x','x'))
                    
                    g.nodes[ntype_1].data['x'], attention_0 = self.decoder_abag[j](g, meta_path_1,\
                                                                (ntype_0,ntype_1),('x','x'))
            
            if self.update_nodes_with_gt_cross_attn:
                if shared_cross_attn:
                
                    feats = self.decoder_abag[j](g, ntype_0, out_etypes[1])
                    g.nodes[ntype_0].data['x'] = feats
                    feats = self.decoder_abag[j](g, ntype_1, out_etypes[0])
                    g.nodes[ntype_1].data['x'] = feats
                
                else:
                
                    feats = self.decoder_agab[j](g, ntype_0, out_etypes[1])
                    g.nodes[ntype_0].data['x'] = feats
                    feats = self.decoder_abag[j](g, ntype_1, out_etypes[0])
                    g.nodes[ntype_1].data['x'] = feats
            
            # 2. Residual update loop-epi edges from nodes
            if not self.apply_attn_to_edges:
                efeats_list = self.update_int_edges[j](g,out_etypes,nkey='x',residual=j>0)
            else:
                #REMOVE IF NOT USED
                g.edges[out_etypes[1]].data['iescore'] = torch.sum(attention_1, 1)
                g.edges[out_etypes[0]].data['iescore'] = torch.sum(attention_0, 1)
                efeats_list = self.update_int_edges[j](g,out_etypes,nkey='x',
                                                       residual=j>0,
                                                       attn=True,
                                                       akey='iescore')

            for et,ef in zip(out_etypes,efeats_list):
                g.edges[et].data['y'] = ef
            
            if debug:
                edges_int.append(efeats_list)

            # 4. Update all edges (Hetero->Homo->Hetero)
            #Pre-norm before combining?
            efeats_list = self.update_all_edges[j](g)
            for (s,et,d), efeat in zip(g.canonical_etypes,efeats_list):
                g.edges[et].data['y'] = efeat

            if debug:
                edges_all.append(efeats_list)

            if self.update_edge_from_edge:
                efeats_list = self.update_all_edges_from_edges[j](g)
                for (s,et,d), efeat in zip(g.canonical_etypes,efeats_list):
                    g.edges[et].data['y'] = efeat

            if debug:
                edges_all.append(efeats_list)

            # 5. Update nodes from nodes+edges data - GTransformer UniMP paper
            # TODO: maybe 5. alternative: Use MPNN from Ingraham Jaakkola work
            if self.update_edges_in_gt:
                feats_list, attn_ug, efeats_list = self.update_graph[j](g, use_topk=self.use_topk)
            else:
                feats_list, attn_ug = self.update_graph[j](g, use_topk=self.use_topk)
            
            for ntype,feats in zip(g.ntypes, feats_list):
                g.nodes[ntype].data['x'] = feats
            
            if self.update_edges_in_gt:
                for et, ef in zip(g.etypes, efeats_list):
                    g.edges[et].data['y'] = ef


                
            if self.last_all:
                # 6. update edges - Maybe when predicting full graph
                efeats_list = self.update_all_edges_final[j](g)
                for (s,et,d),efeat in zip(g.canonical_etypes,efeats_list):
                    g.edges[et].data['y'] = efeat

                if debug:
                    edges_all.append(efeats_list)
            else:
                #6. New update loop-epi edges only
                efeats_list = self.update_int_edges_final[j](g,out_etypes,nkey='x')
                for et,ef in zip(out_etypes,efeats_list):
                    g.edges[et].data['y'] = ef
                if debug:
                    edges_int.append(efeats_list)

            if self.update_edge_from_edge:
                    efeats_list = self.update_all_edges_from_edges_final[j](g)
                    for (s,et,d), efeat in zip(g.canonical_etypes,efeats_list):
                        g.edges[et].data['y'] = efeat
            
        if debug:
            return g, edges_int, edges_all, attn_ug
        return g

class NEGBlock(nn.Module):
    """
    """
    def __init__(self,
                n_hidden,
                n_heads,
                n_layers,
                dropout=0.15,
                blocks=3,
                num_dist_bins_int=10,
                last_all=False,
                add_pe=True,
                shared_cross_attn=True,
                cross_path_only=False,
                dropout_last=0.20,
                multilayer_gt=False
                ):
        super(NEGBlock,self).__init__()

        #Decoder
        self.n_hidden = n_hidden
        self.blocks = blocks
        self.last_all = last_all
        self.add_pe = add_pe
        #if self.add_pe:
            #lnorm_and_dropout = nn.Sequential(nn.LayerNorm(n_hidden),
                                                            #nn.Dropout(dropout)
                                                            #)
            #self.lnorm_and_drop_pe_ab = nn.ModuleList([lnorm_and_dropout for _ in range(self.blocks)])
            #self.lnorm_and_drop_pe_ag = nn.ModuleList([lnorm_and_dropout for _ in range(self.blocks)])
        
        self.shared_cross_attn = shared_cross_attn
        self.cross_path_only = cross_path_only

        if shared_cross_attn:
            #reuse/share parameters -> should be default for ppi
            self.decoder_abag = nn.ModuleList([HeteroGATConv((n_hidden,n_hidden),n_hidden,n_heads,n_layers,dropout) \
                                                                            for _ in range(self.blocks)])
        
        else:
            self.decoder_abag = nn.ModuleList([HeteroGATConv((n_hidden,n_hidden),n_hidden,n_heads,n_layers,dropout) \
                                                                            for _ in range(self.blocks)])
            self.decoder_agab = nn.ModuleList([HeteroGATConv((n_hidden,n_hidden),n_hidden,n_heads,n_layers,dropout) \
                                                                            for _ in range(self.blocks)])

        self.shared_decoder_homo = nn.ModuleList([HomoGAT(n_hidden,n_layers,n_heads,dropout) for _ in range(self.blocks)])
        self.update_int_edges = nn.ModuleList([EdgeNetLayer(n_hidden, n_hidden, dropout) for _ in range(self.blocks)])
        self.update_all_edges = nn.ModuleList([EdgeNetHomoLayer(n_hidden, n_hidden, dropout) for _ in range(self.blocks)])
        if not multilayer_gt:
            self.update_graph = nn.ModuleList([GTransformerHomo(n_hidden,n_heads, dropout)
                                                 for _ in range(self.blocks)])
        else:
            self.update_graph = nn.ModuleList([MultiLayerHomoGT(n_hidden, n_heads, 2,
                                                                dropout=dropout)
                                                for _ in range(self.blocks)])
        if self.last_all:
            self.update_all_edges_final = nn.ModuleList([
                                            EdgeNetHomoLayer(n_hidden, n_hidden, dropout_last)
                                                for _ in range(self.blocks)])
        else:
            self.update_int_edges_final = nn.ModuleList([
                                            EdgeNetLayer(n_hidden, n_hidden, dropout_last)
                                                for _ in range(self.blocks)])


    def forward(self,g, out_etypes = ['ab-ag', 'ag-ab'],
                in_etypes = ['ab-ab', 'ag-ag'], debug=False):

        #print('ne ',g.num_edges)
        edges_int = []
        edges_all = []
        for j in range(self.blocks):
            
            if self.add_pe:
                for ntype in g.ntypes:
                    g.nodes[ntype].data['x'] =  g.nodes[ntype].data['x'] + \
                                                 g.nodes[ntype].data['pe']
            # 1. Cross Attention Nodes GAT
            if not self.cross_path_only:
                meta_path_1 = [out_etypes[0], in_etypes[1]]
                meta_path_0 = [out_etypes[1], in_etypes[0]]
            else:
                meta_path_1 = [out_etypes[0]]
                meta_path_0 = [out_etypes[1]]

            if self.shared_cross_attn:
                ntype_0, ntype_1 = g.ntypes[0], g.ntypes[1]
                g.nodes[ntype_0].data['x'], attention = self.decoder_abag[j](g,meta_path_0,\
                                                            (ntype_1, ntype_0),('x','x'))
                g.nodes[ntype_1].data['x'], attention = self.decoder_abag[j](g, meta_path_1,\
                                                            (ntype_0,ntype_1),('x','x'))
            else:
                g.nodes[ntype_0].data['x'], attention = self.decoder_agab[j](g, meta_path_0,
                                                        (ntype_1, ntype_0),('x','x'))
                
                g.nodes[ntype_1].data['x'], attention = self.decoder_abag[j](g, meta_path_1,\
                                                            (ntype_0,ntype_1),('x','x'))


            # 2. Residual update loop-epi edges from nodes
            efeats_list = self.update_int_edges[j](g,out_etypes,nkey='x',residual=j>0)
            for et,ef in zip(out_etypes,efeats_list):
                g.edges[et].data['y'] = ef
            
            if debug:
                edges_int.append(efeats_list)

            # 3. Update all nodes(Hetero->Homo->Hetero)
            feats_list = self.shared_decoder_homo[j](g)
            for ntype,feats in zip(g.ntypes, feats_list):
                g.nodes[ntype].data['x'] = feats

            # 4. Update all edges (Hetero->Homo->Hetero)
            efeats_list = self.update_all_edges[j](g)
            for (s,et,d),efeat in zip(g.canonical_etypes,efeats_list):
                g.edges[et].data['y'] = efeat

            


            if debug:
                edges_all.append(efeats_list)

            # TODO: maybe 5. alternative: Use MPNN from Ingraham Jaakkola work
            if debug:
                feats_list, attn_ug = self.update_graph[j](g, debug=debug)
            else:
                feats_list = self.update_graph[j](g)
            
            for ntype,feats in zip(g.ntypes, feats_list):
                g.nodes[ntype].data['x'] = feats

            if self.last_all:
                # 6. update edges - Maybe when predicting full graph
                efeats_list = self.update_all_edges_final[j](g)
                for (s,et,d),efeat in zip(g.canonical_etypes,efeats_list):
                    g.edges[et].data['y'] = efeat
                if debug:
                    edges_all.append(efeats_list)
            else:
                #6. New update loop-epi edges only
                efeats_list = self.update_int_edges_final[j](g,out_etypes,nkey='x')
                for et,ef in zip(out_etypes,efeats_list):
                    g.edges[et].data['y'] = ef
                if debug:
                    edges_int.append(efeats_list)
            
        if debug:
            return g, edges_int, edges_all, attn_ug
        return g
