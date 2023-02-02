# 使用关键代码行序列、SBT序列来生成代码注释
import torch
import torch.nn as nn
import config
from .encoder import Encoder
from .decoder import Decoder
import torch.nn.functional as F
import random

class MainModel(nn.Module):
  def __init__(self, keycode_vcblry_size, nl_vcblry_size, sbt_vcblry_size, load_path=None,
            load_dict=None, is_eval=False):
    super(MainModel, self).__init__()
    self.nl_vcblry_size = nl_vcblry_size
    self.is_eval = is_eval

    self.keycode_encoder = Encoder(keycode_vcblry_size)
    self.sbt_encoder = Encoder(sbt_vcblry_size)
    self.hidden_merge_layer = nn.Linear(2*config.encoder_hdn_size, config.encoder_hdn_size)
    self.decoder = Decoder(self.nl_vcblry_size, is_pretrain=False)

    self.hidden_merge_layer.weight.data.normal_(std=1e-4)
    if self.hidden_merge_layer.bias is not None:
      self.hidden_merge_layer.bias.data.normal_(std=1e-4)

    if config.use_cuda:
      self.keycode_encoder = self.keycode_encoder.cuda(config.cuda_id)
      self.sbt_encoder = self.sbt_encoder.cuda(config.cuda_id)
      self.hidden_merge_layer = self.hidden_merge_layer.cuda(config.cuda_id)
      self.decoder = self.decoder.cuda(config.cuda_id)

    if load_path or load_dict:
      state_dict = torch.load(load_path) if not load_dict else load_dict

      self.keycode_encoder.load_state_dict(state_dict['keycode_encoder'])
      self.sbt_encoder.load_state_dict(state_dict['sbt_encoder'])
      self.hidden_merge_layer.load_state_dict(state_dict['hidden_merge_layer'])
      self.decoder.load_state_dict(state_dict['decoder'])

    if self.is_eval:
      self.keycode_encoder.eval()
      self.sbt_encoder.eval()
      self.hidden_merge_layer.eval()
      self.decoder.eval()
    
  def forward(self, batch_data, nl_bos_idx, is_test=False):
    keycode_batch_data, keycode_seq_lens, sbt_batch_data, sbt_seq_lens, \
          nl_batch_data, nl_seq_lens = batch_data
    keycode_enc_opt, keycode_enc_hdn = self.keycode_encoder(keycode_batch_data, keycode_seq_lens)
    sbt_enc_opt, sbt_enc_hdn = self.sbt_encoder(sbt_batch_data, sbt_seq_lens)
    # fuse =torch.cat([keycode_enc_hdn[1:], sbt_enc_hdn[1:]], dim=2)
    fuse = torch.cat([keycode_enc_hdn[:1], sbt_enc_hdn[:1]], dim=2)
    last_dec_hdn = self.hidden_merge_layer(fuse)
    # last_dec_hdn = self.hidden_merge_layer(torch.cat([keycode_enc_hdn[:1], keycode_enc_hdn[:1]], dim=2))
    last_dec_hdn = F.relu(last_dec_hdn)

    if is_test:
      return keycode_enc_opt, sbt_enc_opt, last_dec_hdn

    max_dec_step = max(nl_seq_lens)
    cur_batch_size = len(keycode_seq_lens)
    dec_input = torch.tensor([nl_bos_idx]*cur_batch_size, device=config.device)
    dec_output = torch.zeros((max_dec_step, cur_batch_size, self.nl_vcblry_size), device=config.device)

    for cur_step in range(max_dec_step):
      cur_dec_output, last_dec_hdn = self.decoder(dec_input, last_dec_hdn, keycode_enc_opt, sbt_enc_opt)
      dec_output[cur_step] = cur_dec_output

      if config.use_teacher_forcing and random.random() < config.teacher_forcing_ratio and not self.is_eval:
        dec_input = nl_batch_data[cur_step]
      else:
        _, indices = cur_dec_output.topk(1)
        dec_input = indices.squeeze(1).detach().to(config.device)

    return dec_output
