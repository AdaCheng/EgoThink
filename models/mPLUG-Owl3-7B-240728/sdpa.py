from torch import nn
from icecream import ic
from einops import rearrange

class ScaleDotProductAttention(nn.Module):
    
    def __init__(self, layer_number, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.layer_number = layer_number
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

        # Qwen 不需要scale

    def forward(self, q, k, v, attn_mask=None, order='sbhd'):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        # (N,...,L,E)
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        if order == 'sbhd':
            q, k, v = [rearrange(x, 's b h d -> b h s d').contiguous()
                       for x in (q, k, v)]
        elif order == 'bhsd':
            pass

        if attn_mask is not None:
            attn_mask = (~attn_mask.clone().bool()).contiguous()
        else:
            attn_mask = None
        # attention mask, True means it will take part in attention B H s_q s_k
        if self.training:
            # during training q,k,v always have same seqlen
            if self.causal:
                assert q.shape[-2] == k.shape[-2]
            is_causal = self.causal
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            if self.causal:
                is_causal = q.shape[-2] == k.shape[-2]
            else:
                is_causal = self.causal
            dropout_p = 0.0

        # 如果is_causal则无视输入的mask 反之会使用输入的mask
        o = F.scaled_dot_product_attention(q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=dropout_p, 
            is_causal=is_causal, 
            scale=self.softmax_scale
            )
        # B Head L D -> L B (Head D)
        o = rearrange(o, 'B Head L D -> L B (Head D)').contiguous()
        return o