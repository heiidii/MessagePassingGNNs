import torch
import math
import dgl


def reduce_topk(in_key, o_key, topk):
    def func_reduce(nodes):
        msgs = nodes.mailbox[in_key]
        k = min(msgs.shape[1], topk)
        msg_topk, _ = torch.topk(msgs, k, dim=1)
        o = torch.sum(msg_topk, dim=1)
        return {o_key: o}
    return func_reduce

def reduce_topk_min(in_key, o_key, topk):
    def func_reduce(nodes):
        msgs = nodes.mailbox[in_key]
        k = min(msgs.shape[1], topk)
        #Get max attn of msgs
        msgs, _ = msgs.max(dim=2)
        #get edges with largest max attn
        msg_topk, _ = torch.topk(msgs, k, dim=1)
        msg_min, _ = torch.min(msg_topk, 1)
        return {o_key: msg_min}
    return func_reduce

def apply_topk_edges(o_key, d, eps=1e-8):
    def func_edges(edges):
        msg = edges.data['escore']
        msg_dst_min = edges.dst['escore_min']
        
        x = (1/math.sqrt(d)) * msg / (eps + edges.dst['z'])
        m, _ = torch.max(msg, dim=1)
        select_t = torch.ge(m, msg_dst_min)
        select_t = select_t.unsqueeze(1).expand(-1, x.shape[1], -1)
        x[select_t] = 0.0
        
        return {o_key: x}
    return func_edges
