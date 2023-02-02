from datetime import datetime
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import *
from beam_search import beam_search_sents
from models.pretrain_model import PreTrainModel
from models.main_model import MainModel
import nltk
import config


def test(model_state_dict, vcblry_size, test_loader, nl_vcblry, dataset_size, is_main_model, logger):
    if not is_main_model:
        test_model = PreTrainModel(vcblry_size[0], vcblry_size[1], load_dict=model_state_dict, is_eval=True)
    else:
        test_model = MainModel(vcblry_size[0], vcblry_size[1], vcblry_size[2], load_dict=model_state_dict, is_eval=True)

    total_s_bleu = 0
    total_nl_batch = []
    total_cands = []
    for b_idx, batch in tqdm(enumerate(test_loader)):
        nl_batch = batch[4]
        batch_size = batch[0].shape[1]
        with torch.no_grad():
            if not is_main_model:
                keycode_enc_opt, dec_hidden = test_model(batch, nl_vcblry.word2idx['<BOS>'], is_test=True)
                batch_idx_seqs = beam_search_sents(test_model, nl_vcblry, batch_size, keycode_enc_opt, dec_hidden)
            else:
                keycode_enc_opt, sbt_enc_opt, dec_hidden = test_model(batch, nl_vcblry.word2idx['<BOS>'], is_test=True)
                batch_idx_seqs = beam_search_sents(test_model, nl_vcblry, batch_size, keycode_enc_opt, dec_hidden,
                                                   sbt_enc_opt)

            candidates = trans_idx2word(batch_idx_seqs, nl_vcblry)
            batch_s_bleu = cal_batch_sentence_bleu(nl_batch, candidates)

        total_s_bleu += batch_s_bleu
        total_nl_batch += nl_batch
        total_cands += candidates

    with open(config.dataset_base_path + '/preds_seq_fuse_enc2.txt', 'w', encoding='utf-8') as fp, \
            open(config.dataset_base_path + '/ref_seq_fuse_enc2.txt', 'w', encoding='utf-8') as fr:
        for pred, ref in zip(total_cands, total_nl_batch):
            fp.write(' '.join(pred) + '\n')
            fr.write(' '.join(ref) + '\n')

    s_bleu = total_s_bleu / dataset_size
    c_bleu = cal_corpus_bleu_score(total_nl_batch, total_cands)
    logger.info('S_BLEU: {:.4f}, C_BLEU: {:.4f}.'.format(s_bleu, c_bleu))


def cal_batch_sentence_bleu(refs, cands):
    batch_s_bleu = 0
    smoothing_function = SmoothingFunction()
    for idx in range(len(refs)):
        if len(cands[idx]) != 1:
            cur_bleu = nltk.translate.bleu([refs[idx]], cands[idx], smoothing_function=smoothing_function.method4)
            batch_s_bleu += cur_bleu
    return batch_s_bleu


def cal_corpus_bleu_score(refs, cands):
    smoothing_function = SmoothingFunction()
    return corpus_bleu(list_of_references=[[ref] for ref in refs], hypotheses=[cand for cand in cands],
                       smoothing_function=smoothing_function.method5)


def trans_idx2word(batch_idx_seqs, nl_vcblry):
    word_seqs = []
    for idx_seq in batch_idx_seqs:
        seq = []
        for idx in idx_seq:
            word = nl_vcblry.idx2word[idx]
            if not is_spec_token(word):
                seq.append(word)
        word_seqs.append(seq)
    return word_seqs


def is_spec_token(word):
    if word in ['<BOS>', '<EOS>', '<PAD>']:
        return True
    return False

