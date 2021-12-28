import torch.nn as nn
import torch
import dgl
import dgl.function as fn
import torch.nn.functional as F
import math

from dgl_mp_functions import apply_topk_edges, reduce_topk, reduce_topk_min

class GTransformerHomo(nn.Module):
    '''
    https://arxiv.org/pdf/2009.03509.pdf
    Also compared to pytorch geometric implementation.
    Matches well.
    '''
    def __init__(self, model_dim, n_heads, dropout=0.15, update_edges=False):
        super(GTransformerHomo, self).__init__()

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
        self.dropout = nn.Dropout(dropout)
        self.update_edges = update_edges
        if update_edges:
            self.linears.update((dict(er=nn.Linear(model_dim, model_dim))))
            self.linear_ge = nn.Linear(model_dim*3, model_dim)
            self.linear_gel = nn.Linear(self.d*3, model_dim)
            self.act_e = nn.ReLU()
            self.lnorm_e = nn.LayerNorm(model_dim)
            self.lnorm_ge = nn.LayerNorm(model_dim)

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
            return {okey: self.linears[okey](y).view(y.shape[0],
                                                      self.h,
                                                      self.d)
                    }
        return func

    def propagate_attention(self, g, nkey, ekey, aggregate='concat',
                            eps=1e-8, topk=60, use_topk=False):

        g.apply_nodes(self.func_nodes(nkey, 'q'))
        g.apply_nodes(self.func_nodes(nkey, 'k'))
        g.apply_edges(self.func_edges(ekey, 'e'))
        g.apply_edges(fn.u_add_e('k', 'e', 'ke'))
        g.apply_edges(fn.v_dot_e('q', 'ke', 'score'))

        #print(g.ndata['q'].shape, g.ndata['k'].shape,
        #      g.edata['e'].shape, g.edata['ke'].shape,
        #      g.edata['score'].shape)
        
        g.apply_edges(lambda edges: {'escore': 
                                      torch.exp(edges.data['score'])
                                    })
        
        if not use_topk:
            g.send_and_recv(g.edges(), fn.copy_e('escore', 'escore'),
                        fn.sum('escore', 'z'))
            g.apply_edges(lambda edges: {'norm_escore':
                                     self.dropout(
                                         (1/math.sqrt(self.d))*
                                         edges.data['escore'] 
                                     / (eps + edges.dst['z']))})
        else:
            g.send_and_recv(g.edges(), fn.copy_e('escore', 'escore'),
                            reduce_topk('escore', 'z', topk))
            
            g.send_and_recv(g.edges(), fn.copy_e('escore', 'escore'),
                            reduce_topk_min('escore', 'escore_min', topk))
            
            g.apply_edges(apply_topk_edges('norm_escore', self.d))
            g.apply_edges(lambda edges: {'norm_escore': self.dropout(edges.data['norm_escore'])})
            
        g.apply_nodes(self.func_nodes(nkey, 'v'))
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
            # normalized already (hence not dividing by z)
            g.apply_nodes(lambda nodes: {'o': nodes.data['o']})
        else:
            g.apply_nodes(lambda nodes: {'o':
                                         torch.sum(nodes.data['o'],
                                                   dim=1).squeeze(1)
                                         })
        if not self.update_edges:
            return g.ndata['o'], g.edata['norm_escore']
        else:
            return g.ndata['o'], g.edata['norm_escore'], g.edata['o']

    def forward(self, g_in, nkey='x', ekey='y', aggregate='concat',\
                use_topk=False):

        g = dgl.to_homogeneous(g_in, ndata=[nkey], edata=[ekey])
        if not self.update_edges:
            h_update, attention = self.propagate_attention(g, nkey, ekey,
                                                           aggregate=aggregate,
                                                           use_topk=use_topk)
        else:
            h_update, attention, e_update = self.propagate_attention(g, nkey, ekey,
                                                                     aggregate=aggregate,
                                                                     use_topk=use_topk)
        
        if aggregate == 'concat':
            h_update = h_update.view(-1, self.d*self.h)
            g.apply_nodes(self.func_nodes(nkey, 'r'))
            residual = g.ndata['r'].view(-1, self.d*self.h)
            cat_hr = torch.cat([h_update, residual, h_update-residual], dim=1)
            b = self.sigmoid(self.linear_g(cat_hr)).view(-1, self.d*self.h)
            g.ndata['x'] = F.leaky_relu(self.lnorm((h_update - b*h_update
                                                    + b*residual)))
            
            g.edata['attn'] = attention
            
            if self.update_edges:
            
                e_update = e_update.view(-1, self.d*self.h)
                g.apply_edges(self.func_edges(ekey, 'er'))
                residual = g.edata['er'].view(-1, self.d*self.h)
                cat_er = torch.cat([e_update, residual, e_update-residual], dim=1)
                b = self.sigmoid(self.linear_ge(cat_er)).view(-1, self.d*self.h)
            
                g.edata['y'] = self.act_e(self.lnorm_ge(e_update - b*e_update
                                                        + b*residual))
        
        else:

            g.apply_nodes(self.func_nodes_last(nkey, 'rl'))
            h_update = self.propagate_attention(g, nkey, ekey,
                                                aggregate=aggregate)
            residual = g.ndata['rl']
            cat_hr = torch.cat([h_update, residual, h_update-residual], dim=1)
            b = self.sigmoid(self.linear_gl(cat_hr))
            g.ndata['x'] = h_update - b*h_update + b*residual
            
            g.edata['attn'] = attention
            
            if self.update_edges:
                g.apply_edges(self.func_edges(ekey, 'erl'))
                residual = g.edata['erl']
                cat_er = torch.cat([e_update, residual, e_update-residual], dim=1)
                b = self.sigmoid(self.linear_ge(cat_er))
                g.edata['y'] = self.act_e(self.lnorm_ge(e_update - b*e_update
                                                        + b*residual))

        hetero_g = dgl.to_heterogeneous(g, g_in.ntypes, g_in.etypes)
        list_output = []
        for ntype in g_in.ntypes:
            list_output.append(hetero_g.nodes[ntype].data[nkey])

        if self.update_edges:
            list_output_e = []
            for etype in g_in.etypes:
                list_output_e.append(hetero_g.edges[etype].data['y'])
        
        list_attn = []
        for etype in g_in.etypes:
            list_attn.append(hetero_g.edges[etype].data['attn'])
        
        if not self.update_edges:
            return list_output, list_attn
        else:
            return list_output, list_attn, list_output_e

