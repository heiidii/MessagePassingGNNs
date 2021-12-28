from torch import tensor
import torch.nn as nn
import torch
import dgl.function as fn
import torch.nn.functional as F
import math
import dgl
from dgl_mp_functions import apply_topk_edges, reduce_topk, reduce_topk_min

class GTransformerHetero(nn.Module):
    '''
    https://arxiv.org/pdf/2009.03509.pdf
    Also compared to pytorch geometric implementation.
    Matches well.
    '''
    def __init__(self, model_dim, n_heads, dropout=0.15, update_edges=False):
        super(GTransformerHetero, self).__init__()

        self.d = model_dim // n_heads
        self.h = n_heads
        self.linears = nn.ModuleDict({
                            'q': nn.Linear(model_dim, model_dim),
                            'k': nn.Linear(model_dim, model_dim),
                            'v': nn.Linear(model_dim, model_dim),
                            'e': nn.Linear(model_dim, model_dim),
                            'r': nn.Linear(model_dim, model_dim),
                            'rl': nn.Linear(model_dim, model_dim // n_heads)
        })
        self.linear_g = nn.Linear(model_dim*3, model_dim)
        self.linear_gl = nn.Linear(self.d*3, model_dim)
        self.sigmoid = nn.Sigmoid()
        self.lnorm = nn.LayerNorm(model_dim)
        self.act = nn.ReLU()
        self.leakyact = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.update_edges = update_edges
        if update_edges:
            self.linear_ge = nn.Linear(model_dim*3, model_dim)
            self.linear_gel = nn.Linear(self.d*3, model_dim)
            self.act_e = nn.ReLU()
            self.lnorm_e = nn.LayerNorm(model_dim)


    def func_nodes(self, nkey, okey):
        def func(nodes):
            x = nodes.data[nkey]
            return {okey: self.linears[okey](x).view(x.shape[0], self.h, self.d)}
        return func

    def func_nodes_last(self, nkey, okey):
        def func(nodes):
            x = nodes.data[nkey]
            return {okey: self.linears[okey](x)}
        return func

    def func_edges(self, ekey, okey):
        def func(edges):
            y = edges.data[ekey]
            return {okey: self.linears[okey](y).view(y.shape[0], self.h, self.d)}
        return func


    def propagate_attention(self, g, ntype_src, ntype_dst, etype, nkey, ekey,\
                            aggregate='concat', eps=1e-8,
                            topk=30, use_topk=False):
        
        # for ab-ab, ag-ag edges in a heterograph
        # if want to use it for ag-ab type of edges, additonally specify
        # nodes for ag-ab edges only - see how to do this in GATComplex model
        g.apply_nodes(self.func_nodes(nkey, 'q'), ntype=ntype_dst)
        g.apply_nodes(self.func_nodes(nkey, 'k'), ntype=ntype_src)
        g.apply_edges(self.func_edges(ekey, 'e'))
        g.apply_edges(fn.u_add_e('k', 'e', 'ke'))
        g.apply_edges(fn.v_dot_e('q', 'ke', 'score'))
        g.apply_edges(lambda edges: {'escore':
                                      torch.exp(edges.data['score'])})
        if not use_topk:
            g.send_and_recv(g.edges(), fn.copy_e('escore', 'escore'),
                        fn.sum('escore', 'z'))

            g.apply_edges(lambda edges: {'norm_escore':
                                     self.dropout((1/math.sqrt(self.d))*
                                     edges.data['escore']/(eps + edges.dst['z']))
                                     })
        else:
            g.send_and_recv(g.edges(), fn.copy_e('escore', 'escore'),
                            reduce_topk('escore', 'z', topk))
            
            g.send_and_recv(g.edges(), fn.copy_e('escore', 'escore'),
                            reduce_topk_min('escore', 'escore_min', topk))
            g.apply_edges(apply_topk_edges('norm_escore', self.d))
            g.apply_edges(lambda edges: {'norm_escore': self.dropout(edges.data['norm_escore'])})

        g.apply_nodes(self.func_nodes(nkey, 'v'), ntype=ntype_src)
        g.apply_edges(fn.u_add_e('v', 'e', 've'))
        g.apply_edges(lambda edges: {'vescore':
                                     edges.data['norm_escore']*edges.data['ve']
                                     })
        if self.update_edges:
            g.apply_edges(lambda edges: {'o': 
                                         edges.data['norm_escore']*edges.data['e']})

        g.send_and_recv(g.edges(), fn.copy_e('vescore', 'vescore'),
                        fn.sum('vescore', 'o'))

        if aggregate == 'concat':
            g.apply_nodes(lambda nodes: {'o': nodes.data['o']}, ntype=ntype_dst)
        else:
            g.apply_nodes(lambda nodes: {'o':
                                         torch.sum(nodes.data['o'],
                                                   dim=1).squeeze(1)
                                         },
                          ntype=ntype_dst)
        
        if not self.update_edges:
            return g.nodes[ntype_dst].data['o'], g.edata['norm_escore']
        else:
            return g.nodes[ntype_dst].data['o'], g.edata['norm_escore'],\
                    g.edata['o']
    
    
    def forward(self, fg, ntype, etype, nkey='x', ekey='y',\
                aggregate='concat', debug=False, use_topk=False):
        
        (ntype_src, _, ntype_dst) = fg.to_canonical_etype(etype)
        #Get metapath
        meta_paths = [etype]
        dict_num_nodes_batch = {}
        for nt in fg.ntypes:
            dict_num_nodes_batch[nt] = fg.batch_num_nodes(ntype=nt)

        dict_num_edges_batch = {}
        for et in meta_paths:
            dict_num_edges_batch[et] = fg.batch_num_edges(etype=et)

        g = dgl.metapath_reachable_graph(fg, meta_paths)
        g.set_batch_num_nodes(dict_num_nodes_batch)
        g.set_batch_num_edges(dict_num_edges_batch)
        g.edata[ekey] = fg.edges[etype].data[ekey]
        
        if not self.update_edges:
            h_update, attention = \
                self.propagate_attention(g, ntype_src, ntype_dst, etype, nkey, ekey,
                                         aggregate=aggregate,
                                         use_topk=use_topk)
        else:
            h_update, attention, e_update = \
                self.propagate_attention(g, ntype_src, ntype_dst, etype, nkey, ekey,
                                         aggregate=aggregate,
                                         use_topk=use_topk)

        if aggregate == 'concat':
            h_update = h_update.view(-1, self.d*self.h)
            g.apply_nodes(self.func_nodes(nkey, 'r'), ntype=ntype_dst)
            residual = g.nodes[ntype_dst].data['r'].view(-1, self.d*self.h)
            cat_hr = torch.cat([h_update, residual, h_update-residual], dim=1)
            b = self.sigmoid(self.linear_g(cat_hr)).view(-1, self.d*self.h)
            #final update
            g.nodes[ntype_dst].data[nkey] = \
                self.leakyact(self.lnorm((h_update - b*h_update + b*residual)))

            if self.update_edges:
                #untested
                e_update = e_update.view(-1, self.d*self.h)
                g.apply_edges(self.func_edges(ekey, 'er'), etype=etype)
                residual_e = g.edges[etype].data['er'].view(-1, self.d*self.h)
                cat_er = torch.cat([e_update, residual_e, e_update-residual_e], dim=1)
                b = self.sigmoid(self.linear_ge(cat_er)).view(-1, self.d*self.h)
                g.edges[etype].data[ekey] = self.act_e(
                                                self.lnorm_e((e_update\
                                                            - b*e_update + b*residual_e))
                                                            )
        
        else:
            g.apply_nodes(self.func_nodes_last(nkey, 'rl'), ntype=ntype_dst)
            residual = g.nodes[ntype_dst].data['rl']
            cat_hr = torch.cat([h_update, residual, h_update-residual], dim=1)
            b = self.sigmoid(self.linear_gl(cat_hr))
            g.nodes[ntype_dst].data[nkey] = h_update - b*h_update + b*residual

            if self.update_edges:
                g.apply_edges(self.func_edges(ekey, 'erl'), etype=etype)
                residual_e = g.edges[etype].data['erl']
                cat_er = torch.cat([e_update, residual_e, e_update-residual_e], dim=1)
                b = self.sigmoid(self.linear_gel(cat_er))
                g.edges[etype].data[ekey] = self.act_el(
                                                self.lnorm_el((e_update\
                                                            - b*e_update + b*residual_e))
                                                            )

        if debug:
            if not self.update_edges:
                return g.nodes[ntype_dst].data[nkey], attention
            else:
                return g.nodes[ntype_dst].data[nkey], attention, g.edges[etype].data[ekey]
        else:
            if not self.update_edges:
                return g.nodes[ntype_dst].data[nkey]
            else:
                return g.nodes[ntype_dst].data[nkey], g.edges[etype].data[ekey]

