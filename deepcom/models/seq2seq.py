#!/usr/bin/env python
# @Project ：code 
# @File    ：seq2seq.py
# @Author  ：
# @Date    ：2022/1/24 16:14 
# 
# --------------------------------------------------------
import torch
import torch.nn as nn
import config
import torch.nn.functional as F
import random
import math

attention = []

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

    def forward(self, src, src_length_lst):
        src_emb = self.embedding(src)
        src_packed = nn.utils.rnn.pack_padded_sequence(src_emb, src_length_lst, batch_first=False, enforce_sorted=False)
        output, hidden = self.gru(src_packed)
        output_padded, _length = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        res = output_padded[:, :, :self.hdn_size] + output_padded[:, :, self.hdn_size:]
        return res, hidden


class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.hdn_size = config.decoder_hdn_size

        self.Wa = nn.Linear(2 * self.hdn_size, self.hdn_size)
        self.va = nn.Parameter(torch.rand(self.hdn_size), requires_grad=True)
        self.va.data.normal_(mean=0, std=1 / math.sqrt(self.va.size(0)))

    def forward(self, last_hidden, encoder_outputs):
        seq_lens, batch_size, _ = encoder_outputs.size()
        lh = last_hidden.repeat(seq_lens, 1, 1).transpose(0, 1)
        attention_energies = self.score(lh, encoder_outputs.transpose(0, 1))
        att = F.softmax(attention_energies, dim=1).unsqueeze(1)
        return att

    def score(self, last_hidden, encoder_outputs):
        energies = F.relu(self.Wa(torch.cat([last_hidden, encoder_outputs], dim=2)))
        energies = energies.transpose(1, 2)
        va = self.va.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energies = torch.bmm(va, energies)
        return energies.squeeze(1)


class Decoder(nn.Module):
    def __init__(self, tgt_vcblry_size, is_pretrain=True):
        super(Decoder, self).__init__()
        self.hdn_size = config.decoder_hdn_size
        self.embedding_dim = config.embedding_dim
        self.is_pretrain = is_pretrain

        self.embedding = nn.Embedding(tgt_vcblry_size, self.embedding_dim)
        self.keycode_attention = AttentionLayer()
        if not self.is_pretrain:
            self.sbt_attention = AttentionLayer()
        self.gru = nn.GRU(self.embedding_dim + self.hdn_size, self.hdn_size)
        self.linear = nn.Linear(2 * self.hdn_size, tgt_vcblry_size)
        self.init_paras()

    def init_paras(self):
        init_std = 1e-4
        self.embedding.weight.data.normal_(std=init_std)

        self.linear.weight.data.normal_(std=init_std)
        if self.linear.bias is not None:
            self.linear.bias.data.normal_(std=init_std)

        for wgts in self.gru._all_weights:
            for wgt_name in wgts:
                if wgt_name.startswith('weight_'):
                    wgt = getattr(self.gru, wgt_name)
                    wgt.data.uniform_(-0.02, 0.02)
                elif wgt_name.startswith('bias_'):
                    b = getattr(self.gru, wgt_name)
                    b.data.fill_(0.0)
                    b.data[b.size(0) // 4: b.size(0) // 2].fill_(1.0)

    def forward(self, inputs, last_hidden, keycode_enc_outputs):
        input_embed = self.embedding(inputs).unsqueeze(0)
        keycode_atte_weights = self.keycode_attention(last_hidden, keycode_enc_outputs)

        keycode_context = keycode_atte_weights.bmm(keycode_enc_outputs.transpose(0, 1)).transpose(0, 1)
        context = keycode_context

        output, hidden = self.gru(torch.cat([input_embed, context], dim=2), last_hidden)
        output = self.linear(torch.cat([output.squeeze(0), context.squeeze(0)], dim=1))
        output = F.log_softmax(output, dim=1)
        return output, hidden


class MainModel(nn.Module):
    def __init__(self, keycode_vcblry_size, nl_vcblry_size, load_path=None, load_dict=None, is_eval=False):
        super(MainModel, self).__init__()
        self.nl_vcblry_size = nl_vcblry_size
        self.is_eval = is_eval

        self.keycode_encoder = Encoder(keycode_vcblry_size)
        self.hidden_merge_layer = nn.Linear(2 * config.encoder_hdn_size, config.encoder_hdn_size)
        self.decoder = Decoder(self.nl_vcblry_size, is_pretrain=False)

        self.hidden_merge_layer.weight.data.normal_(std=1e-4)
        if self.hidden_merge_layer.bias is not None:
            self.hidden_merge_layer.bias.data.normal_(std=1e-4)

        if config.use_cuda:
            self.keycode_encoder = self.keycode_encoder.cuda(config.cuda_id)
            self.hidden_merge_layer = self.hidden_merge_layer.cuda(config.cuda_id)
            self.decoder = self.decoder.cuda(config.cuda_id)

        if load_path or load_dict:
            state_dict = torch.load(load_path) if not load_dict else load_dict

            self.keycode_encoder.load_state_dict(state_dict['keycode_encoder'])
            self.hidden_merge_layer.load_state_dict(state_dict['hidden_merge_layer'])
            self.decoder.load_state_dict(state_dict['decoder'])

        if self.is_eval:
            self.keycode_encoder.eval()
            self.hidden_merge_layer.eval()
            self.decoder.eval()

    def forward(self, batch_data, nl_bos_idx, is_test=False):
        keycode_batch_data, keycode_seq_lens, nl_batch_data, nl_seq_lens = batch_data
        keycode_enc_opt, keycode_enc_hdn = self.keycode_encoder(keycode_batch_data, keycode_seq_lens)

        # last_dec_hdn = self.hidden_merge_layer(torch.cat([keycode_enc_hdn[:1], keycode_enc_hdn[:1]], dim=2))
        # last_dec_hdn = F.relu(last_dec_hdn)
        last_dec_hdn = keycode_enc_hdn[:1]

        if is_test:
            return keycode_enc_opt, last_dec_hdn

        max_dec_step = max(nl_seq_lens)
        cur_batch_size = len(keycode_seq_lens)
        dec_input = torch.tensor([nl_bos_idx] * cur_batch_size, device=config.device)
        dec_output = torch.zeros((max_dec_step, cur_batch_size, self.nl_vcblry_size), device=config.device)

        for cur_step in range(max_dec_step):
            cur_dec_output, last_dec_hdn = self.decoder(dec_input, last_dec_hdn, keycode_enc_opt)
            dec_output[cur_step] = cur_dec_output

            if config.use_teacher_forcing and random.random() < config.teacher_forcing_ratio and not self.is_eval:
                dec_input = nl_batch_data[cur_step]
            else:
                _, indices = cur_dec_output.topk(1)
                dec_input = indices.squeeze(1).detach().to(config.device)

        return dec_output
