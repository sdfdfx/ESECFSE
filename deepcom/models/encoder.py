# 关键代码行序列和SBT序列的Encoder
import torch
import torch.nn as nn
import config


class Encoder(nn.Module):
    def __init__(self, src_vcblry_size):
        super(Encoder, self).__init__()
        self.hdn_size = config.encoder_hdn_size
        self.embedding_dim = config.embedding_dim

        self.embedding = nn.Embedding(src_vcblry_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hdn_size, bidirectional=True)
        self.init_paras()

    def init_paras(self):
        self.embedding.weight.data.normal_(std=1e-4)
        for wgts in self.gru._all_weights:
            for wgt_name in wgts:
                if wgt_name.startswith('weight_'):
                    wgt = getattr(self.gru, wgt_name)
                    wgt.data.uniform_(-0.02, 0.02)
                elif wgt_name.startswith('bias_'):
                    b = getattr(self.gru, wgt_name)
                    b.data.fill_(0.0)
                    b.data[b.size(0) // 4: b.size(0) // 2].fill_(1.0)

    def forward(self, src, src_length_lst, fuse=False):
        src_emb = self.embedding(src)
        src_packed = nn.utils.rnn.pack_padded_sequence(src_emb, src_length_lst, batch_first=False, enforce_sorted=False)
        output, hidden = self.gru(src_packed)
        output_padded, _length = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        res = output_padded[:, :, :self.hdn_size] + output_padded[:, :, self.hdn_size:]
        return res, hidden
