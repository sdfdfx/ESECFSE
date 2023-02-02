import torch
import config


class BeamSearchNode(object):
    def __init__(self, sent_idx_lst, log_prbblty, hidden):
        self.sent_idx_lst = sent_idx_lst
        self.log_prbblty = log_prbblty
        self.hidden = hidden

    def append_node(self, word_idx, log_prbblty, hidden):
        return BeamSearchNode(self.sent_idx_lst + [word_idx], self.log_prbblty + [log_prbblty], hidden)

    def get_avg_prbblty(self):
        return sum(self.log_prbblty) / len(self.log_prbblty)

    def get_last_word_idx(self):
        return self.sent_idx_lst[-1]


def beam_search_sents(test_model, nl_vcblry, batch_size, keycode_enc_opt, dec_hidden, sbt_enc_opt=None,
                      cfg_enc_opt=None):
    batch_idx_seqs = []
    for idx in range(batch_size):
        cur_idx_kyc_enc_opt = keycode_enc_opt[:, idx, :].unsqueeze(1)
        cur_idx_dec_hidden = dec_hidden[:, idx, :].unsqueeze(1)
        if sbt_enc_opt != None:
            cur_idx_sbt_enc_opt = sbt_enc_opt[:, idx, :].unsqueeze(1)
        if cfg_enc_opt != None:
            cur_idx_cfg_enc_opt = cfg_enc_opt[:, idx, :].unsqueeze(1)

        start_node = BeamSearchNode([nl_vcblry.word2idx['<BOS>']], [0.], cur_idx_dec_hidden)
        cur_node_lst = [start_node]
        end_node_lst = []

        for step in range(config.max_translate_length):
            if len(cur_node_lst) == 0:
                break
            cand_next_node_lst = []
            next_inputs = []
            next_hidden = []
            tobe_extend_node_lst = []

            for node in cur_node_lst:
                if node.get_last_word_idx() == nl_vcblry.word2idx['<EOS>']:
                    end_node_lst.append(node)
                    if len(end_node_lst) >= config.beam_width:
                        break
                    else:
                        continue

                tobe_extend_node_lst.append(node)

                next_inputs.append(node.get_last_word_idx())
                next_hidden.append(node.hidden.clone().detach())

            if len(tobe_extend_node_lst) == 0:
                break

            input_size = len(next_inputs)
            next_kyc_enc_opt = cur_idx_kyc_enc_opt.repeat(1, input_size, 1)
            if sbt_enc_opt != None:
                next_sbt_enc_opt = cur_idx_sbt_enc_opt.repeat(1, input_size, 1)
            if cfg_enc_opt != None:
                next_cfg_enc_opt = cur_idx_cfg_enc_opt.repeat(1, input_size, 1)
            next_inputs = torch.tensor(next_inputs, device=config.device)
            next_hidden = torch.stack(next_hidden, dim=2).squeeze(0)

            if sbt_enc_opt == None:
                decoder_opts, last_dec_hidden = test_model.decoder(next_inputs, next_hidden, next_kyc_enc_opt)
            elif cfg_enc_opt == None:
                decoder_opts, last_dec_hidden = test_model.decoder(next_inputs, next_hidden,
                                                                   next_kyc_enc_opt, next_sbt_enc_opt)
            else:
                decoder_opts, last_dec_hidden = test_model.decoder(next_inputs, next_hidden,
                                                                   next_kyc_enc_opt, next_sbt_enc_opt, next_cfg_enc_opt)

            topk_log_prbblty_lst, topk_word_idx_lst = decoder_opts.topk(config.beam_width)
            for idx, node in enumerate(tobe_extend_node_lst):
                cur_topk_log_prbblty = topk_log_prbblty_lst[idx]
                cur_topk_word_idx = topk_word_idx_lst[idx]
                cur_hidden = last_dec_hidden[:, idx, :].unsqueeze(1)

                for i in range(config.beam_width):
                    log_prbblty = cur_topk_log_prbblty[i]
                    word_idx = cur_topk_word_idx[i].item()

                    new_node = node.append_node(word_idx, log_prbblty, cur_hidden)
                    cand_next_node_lst.append(new_node)

            cand_next_node_lst = sorted(cand_next_node_lst, key=lambda x: x.get_avg_prbblty(), reverse=True)
            cur_node_lst = cand_next_node_lst[:config.beam_width]

        end_node_lst += cur_node_lst
        end_node_lst = sorted(end_node_lst, key=lambda x: x.get_avg_prbblty(), reverse=True)
        res_node = end_node_lst[0]

        batch_idx_seqs.append(res_node.sent_idx_lst)

    return batch_idx_seqs
