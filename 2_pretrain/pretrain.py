import torch
from torch import nn
from d2l import torch as d2l

class CheTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=15, key_size=32, query_size=32, value_size=32,
                 hid_in_features=32, mlm_in_features=32,
                 sgp_in_features=32):
        super(CheTModel, self).__init__()
        self.encoder = CheTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.sgp = SpaceGroupPred(sgp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        sgp_Y_hat = self.sgp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, sgp_Y_hat