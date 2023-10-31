import torch
from torch import nn
from d2l import torch as d2l

class CheTEncoder(nn.Module):
    """CheT Encoder."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout=0.2,
                 max_len=100, key_size=32, query_size=32, value_size=32,
                 **kwargs):
        super(CheTEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.element_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.proportion_embedding = nn.Linear(1, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))

    def forward(self, element_numbers, proportions, valid_lens):
        original_shape = proportions.shape
        proportions = proportions.view(-1, 1)
        X = self.proportion_embedding(proportions)
        X = X.view(*original_shape[:-1], -1,self.num_hiddens)
        X += self.element_embedding(element_numbers)
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X