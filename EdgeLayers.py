import torch.nn as nn
import dgl
import torch
import dgl.function as fn
import math

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, nkey='x'):
        def func(edges):
            h_u = edges.src[nkey]
            h_v = edges.dst[nkey]
            score = self.W(torch.cat([h_u, h_v], 1))
            return {'lscore': score}
        return func

    def forward(self, graph, etype, nkey='x'):
        #print(etype)
        graph.apply_edges(self.apply_edges(nkey=nkey), etype=etype)
        return graph.edges[etype].data['lscore']


class MLPEdgePredictor(nn.Module):
    def __init__(self, n_hidden, n_out, dropout=0.20, pre_norm=False):
        super(MLPEdgePredictor, self).__init__()
        #Same as feedforward of transformer- except elu
        #now same as the OneLayer thing
        self.dense = nn.Sequential(nn.Linear(n_hidden*2, n_hidden),
                                   nn.ELU(inplace=True),
                                   nn.Dropout(dropout),
                                   nn.Linear(n_hidden, n_out)
                                   )
        if pre_norm:
            self.pre_norm = pre_norm
            self.norm = nn.LayerNorm(n_hidden*2)
    
    def apply_edges(self, nkey='x', attn=False, akey='iescore'):
        
        def func(edges):
            h_u = edges.src[nkey]
            h_v = edges.dst[nkey]
            if not attn:
                score = self.dense(torch.cat([h_u, h_v], 1))
            else:
                a = edges.data[akey]

                #debug
                #import matplotlib.pyplot as plt
                #h_u_v_a = a*h_u_v
                #n_h_u_v_a = self.norm(h_u_v_a)
                #fig, axs = plt.subplots(2, 2)
                #a_np = a.detach()
                #huv_np = h_u_v.detach()
                #h_u_v_a = h_u_v_a.detach()
                #n_h_u_v_a = n_h_u_v_a.detach()
                #a_np = a_np.expand(-1, 10)
                #im = axs[0,0].imshow(a_np.numpy(), aspect='auto')
                #fig.colorbar(im, ax=axs[0,0], orientation='vertical')
                #im = axs[0,1].imshow(huv_np.numpy(), aspect='auto')
                #fig.colorbar(im, ax=axs[0,1], orientation='vertical')
                #im = axs[1,0].imshow(h_u_v_a.numpy(), aspect='auto')
                #fig.colorbar(im, ax=axs[1,0], orientation='vertical')
                #im = axs[1,1].imshow(n_h_u_v_a.numpy(), aspect='auto')
                #fig.colorbar(im, ax=axs[1,1], orientation='vertical')
                #plt.show()
                #plt.close()
                #exit()
                h_u_v = torch.cat([h_u, h_v], 1)
                h_u_v = a * h_u_v
                if self.pre_norm:
                    h_u_v = self.norm(h_u_v)
                score = self.dense(h_u_v)
            
            return {'lscore': score}
        return func

    def forward(self, graph, etype=None, nkey='x', attn=False, akey='iescore'):
        #print(etype)
        if etype is not None:
            graph.apply_edges(self.apply_edges(nkey=nkey, attn=attn, akey=akey), etype=etype)
            return graph.edges[etype].data['lscore']
        else:
            graph.apply_edges(self.apply_edges(nkey=nkey, attn=attn, akey=akey))
            return graph.edata['lscore']


class MLPEdgefromEdgePredictor(nn.Module):
    def __init__(self, n_hidden, n_out, dropout=0.20, max_reduce=False):
        super(MLPEdgefromEdgePredictor, self).__init__()
        #Same as feedforward of transformer- except elu
        #now same as the OneLayer thing
        self.max_reduce = max_reduce
        self.norm = nn.LayerNorm(n_hidden)
        self.linear_left = nn.Linear(n_hidden, n_hidden)
        self.linear_right = nn.Linear(n_hidden, n_hidden)
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Sequential(nn.LayerNorm(n_hidden*2),
                                   nn.Linear(n_hidden*2, n_hidden),
                                   nn.ELU(inplace=True),
                                   nn.Dropout(dropout),
                                   nn.Linear(n_hidden, n_out)
                                   )
    
    def apply_edges(self, lkey='aleft', rkey='aright', okey='out'):
        def func(edges):
            h_u = edges.src[lkey]
            h_v = edges.dst[rkey]
            return {okey: self.dense(torch.cat([h_u, h_v], 1))}
        return func

        
    def gate_incoming_edges(self):

        def func_edges(edges):
            y = self.norm(edges.data['y'])
            l_gate = self.sigmoid(self.linear_left(y)) #bias with attention from GT
            r_gate = self.sigmoid(self.linear_right(y))
            return {'right': r_gate, 'left': l_gate}

        return func_edges

    def aggregate_neighbor_edges(self, g, etype='ab-ag'):

        g.apply_edges(self.gate_incoming_edges(), etype=etype)


        if self.max_reduce:
            g.send_and_recv(g[etype].edges(), fn.copy_e('left', 'left'),
                            fn.max('left', 'aleft'), etype=etype)
            g.send_and_recv(g[etype].edges(), fn.copy_e('right', 'right'),
                            fn.max('right', 'aright'), etype=etype)
        else:
            g.send_and_recv(g[etype].edges(), fn.copy_e('left', 'left'),
                            fn.mean('left', 'aleft'), etype=etype)
            g.send_and_recv(g[etype].edges(), fn.copy_e('right', 'right'),
                            fn.mean('right', 'aright'), etype=etype)

        g.apply_edges(self.apply_edges(okey='out'), etype=etype)

        return g.edges[etype].data['out']

    def aggregate_neighbor_edges_homo(self, g):

        g.apply_edges(self.gate_incoming_edges())

        if self.max_reduce:
            g.send_and_recv(g.edges(), fn.copy_e('left', 'left'),
                        fn.max('left', 'aleft'))
            g.send_and_recv(g.edges(), fn.copy_e('right', 'right'),
                        fn.max('right', 'aright'))
        else:
            g.send_and_recv(g.edges(), fn.copy_e('left', 'left'),
                        fn.mean('left', 'aleft'))
            g.send_and_recv(g.edges(), fn.copy_e('right', 'right'),
                        fn.mean('right', 'aright'))

        g.apply_edges(self.apply_edges(okey='out'))

        return g.edata['out']

    def forward(self, graph, etype=None):
        if etype is not None:
            return self.aggregate_neighbor_edges(graph, etype=etype)
        else:
            return self.aggregate_neighbor_edges_homo(graph)

class MLPEdgeOneLayerPredictor(nn.Module):
    def __init__(self, n_hidden, n_out, dropout=0.15):
        super(MLPEdgeOneLayerPredictor, self).__init__()
        self.dense = nn.Sequential(nn.Linear(n_hidden*2,n_hidden),
                                   nn.ELU(inplace=True),
                                   nn.Dropout(dropout),
                                   nn.Linear(n_hidden, n_out)
                                   )
    
    def apply_edges(self, nkey='x'):
        def func(edges):
            h_u = edges.src[nkey]
            h_v = edges.dst[nkey]
            score = self.dense(torch.cat([h_u, h_v], 1))
            return {'lscore': score}
        return func

    def forward(self, graph, etype=None,nkey='x'):
        #print(etype)
        if etype is not None:
            graph.apply_edges(self.apply_edges(nkey=nkey),etype=etype)
            return graph.edges[etype].data['lscore']
        else:
            graph.apply_edges(self.apply_edges(nkey=nkey))
            return graph.edata['lscore']

class EdgeNetSimpleLayer(nn.Module):
    def __init__(self, n_hidden, n_out=None, dropout=0.15):
        
        super(EdgeNetSimpleLayer, self).__init__()
        
        if n_out is None:
            n_out = n_hidden
        
        self.dense = MLPEdgeOneLayerPredictor(n_hidden, n_out)
        #self.norm = nn.BatchNorm1d(n_hidden)
        self.norm = nn.LayerNorm(n_hidden)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, g, etypes, nkey='x', ekey='y', residual=True):

        outlist = []
        for et in etypes:
            efeats = g.edges[et].data[ekey]
            out = self.dense(g, et, nkey=nkey)
            if residual:
                out = self.norm(efeats + self.dropout(out))
            else:
                out = self.norm(self.dropout(out))
            outlist.append(out)
        
        return outlist

class EdgeNetLayer(nn.Module):
    def __init__(self, n_hidden, n_out=None, dropout=0.15):
        
        super(EdgeNetLayer, self).__init__()
        
        if n_out is None:
            n_out = n_hidden
        
        self.dense = MLPEdgePredictor(n_hidden, n_out)
        #self.norm = nn.BatchNorm1d(n_hidden)
        self.norm = nn.LayerNorm(n_hidden)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, g, etypes, nkey='x', ekey='y',
                residual=True, attn=False, akey='iescore'):

        outlist = []
        for et in etypes:
            efeats = g.edges[et].data[ekey]
            out = self.dense(g, et, nkey=nkey, attn=attn, akey=akey)
            if residual:
                out = self.norm(efeats + self.dropout(out))
            else:
                out = self.norm(out)
            outlist.append(out)
        
        return outlist


class EdgeNetfromEdgeLayer(nn.Module):
    def __init__(self, n_hidden, n_out=None, dropout=0.15):
        
        super(EdgeNetfromEdgeLayer, self).__init__()
        
        if n_out is None:
            n_out = n_hidden
        
        self.dense = MLPEdgefromEdgePredictor(n_hidden, n_out)
        self.norm = nn.LayerNorm(n_hidden)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, g, etypes, ekey='y',\
                residual=True):

        outlist = []
        for et in etypes:
            efeats = g.edges[et].data[ekey]
            out = self.dense(g, et)
            if residual:
                out = self.norm(efeats + self.dropout(out))
            else:
                out = self.norm(out)
            outlist.append(out)
        
        return outlist



class EdgeNetSimpleHomoLayer(nn.Module):
    def __init__(self, n_hidden, n_out=None, dropout=0.15):
        
        super(EdgeNetSimpleHomoLayer, self).__init__()
        
        if n_out is None:
            n_out = n_hidden
        
        self.dense = MLPEdgeOneLayerPredictor(n_hidden, n_out)
        #self.norm = nn.BatchNorm1d(n_hidden)
        self.norm = nn.LayerNorm(n_hidden)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, g, nkey='x', ekey='y', residual=True):

        homo_g = dgl.to_homogeneous(g,ndata=[nkey],edata=[ekey])
        efeats = homo_g.edata[ekey]
        out = self.dense(homo_g)
        
        if residual:
            out = self.norm(efeats + self.dropout(out))
        else:
            out = self.norm(out)
        
        homo_g.edata[ekey] = out

        hetero_g = dgl.to_heterogeneous(homo_g, g.ntypes, g.etypes)

        output = [ hetero_g.edges[et].data[ekey] \
                    for _,et,_ in hetero_g.canonical_etypes]
        
        return output

class EdgeNetHomoLayer(nn.Module):

    def __init__(self, n_hidden, n_out=None, dropout=0.15):
        
        super(EdgeNetHomoLayer, self).__init__()

        if n_out is None:
            n_out = n_hidden

        self.dense = MLPEdgePredictor(n_hidden, n_out)
        self.norm = nn.LayerNorm(n_hidden)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, g, nkey='x', ekey='y', residual=True):

        homo_g = dgl.to_homogeneous(g, ndata=[nkey], edata=[ekey])
        efeats = homo_g.edata[ekey]
        out = self.dense(homo_g)

        if residual:
            out = self.norm(efeats + self.dropout(out))
        else:
            out = self.norm(out)
        
        homo_g.edata[ekey] = out

        hetero_g = dgl.to_heterogeneous(homo_g, g.ntypes, g.etypes)

        output = [hetero_g.edges[et].data[ekey]
                    for _, et, _ in hetero_g.canonical_etypes]
        
        return output

class EdgeNetfromEdgeHomoLayer(nn.Module):

    def __init__(self, n_hidden, n_out=None, dropout=0.15):
        
        super(EdgeNetfromEdgeHomoLayer, self).__init__()

        if n_out is None:
            n_out = n_hidden

        self.efe = MLPEdgefromEdgePredictor(n_hidden, n_out)
        self.norm = nn.LayerNorm(n_hidden)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, g, nkey='x', ekey='y', residual=True):

        homo_g = dgl.to_homogeneous(g, ndata=[nkey], edata=[ekey])
        out = self.efe(homo_g)

        homo_g.edata[ekey] = out

        hetero_g = dgl.to_heterogeneous(homo_g, g.ntypes, g.etypes)

        output = [hetero_g.edges[et].data[ekey]
                    for _, et, _ in hetero_g.canonical_etypes]
        
        return output


