import torch
import torch.nn as nn
import config
import math
import torch.nn.functional as F

class AttentionLayer(nn.Module):
  def __init__(self):
    super(AttentionLayer, self).__init__()
    self.hdn_size = config.decoder_hdn_size

    self.Wa = nn.Linear(2*self.hdn_size, self.hdn_size)
    self.va = nn.Parameter(torch.rand(self.hdn_size), requires_grad=True)
    self.va.data.normal_(mean=0, std=1/math.sqrt(self.va.size(0)))

  def forward(self, last_hidden, encoder_outputs):
    seq_lens, batch_size, _ = encoder_outputs.size()
    lh = last_hidden.repeat(seq_lens, 1, 1).transpose(0, 1)
    attention_energies = self.score(lh, encoder_outputs.transpose(0, 1))
    return F.softmax(attention_energies, dim=1).unsqueeze(1)

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
    self.gru = nn.GRU(self.embedding_dim+self.hdn_size, self.hdn_size)
    self.linear = nn.Linear(2*self.hdn_size, tgt_vcblry_size)
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
          b.data[b.size(0)//4 : b.size(0)//2].fill_(1.0)

  def forward(self, inputs, last_hidden, keycode_enc_outputs, sbt_enc_outputs=None):
    input_embed = self.embedding(inputs).unsqueeze(0)
    keycode_atte_weights = self.keycode_attention(last_hidden, keycode_enc_outputs)
    keycode_context = keycode_atte_weights.bmm(keycode_enc_outputs.transpose(0, 1)).transpose(0, 1)
    if self.is_pretrain:
      context = keycode_context
    else:
      sbt_atte_weights = self.sbt_attention(last_hidden, sbt_enc_outputs)
      sbt_context = sbt_atte_weights.bmm(sbt_enc_outputs.transpose(0, 1)).transpose(0, 1)
      context = keycode_context + sbt_context
    output, hidden = self.gru(torch.cat([input_embed, context], dim=2), last_hidden)
    output = self.linear(torch.cat([output.squeeze(0), context.squeeze(0)], dim=1))
    output = F.log_softmax(output, dim=1)
    return output, hidden
