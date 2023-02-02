#!/usr/bin/env python
# @Project ：code 
# @File    ：train_fuse_enc.py
# @Author  ：
# @Date    ：2022/9/21 15:28 
# 
# --------------------------------------------------------

import torch.optim as optim
import torch.nn as nn
from models.pretrain_model import PreTrainModel
from models.fuse_encoder import MainModel
import config
import logging
from torch.utils.data import TensorDataset, DataLoader
from vocabulary import Vocabulary
from datetime import datetime
import os
import torch
import itertools
import random
import time
from torch.utils.data import Dataset

from tqdm import tqdm
from nltk.translate.bleu_score import *
from beam_search import beam_search_sents

import nltk


# ------------------------------------------ train --------------------------------------------------------------#
def train(vcblry_size, train_loader, valid_loader, nl_vcblry, dataset_size, is_main_model, logger,
          pretrain_kyc_enc=None):
    if not is_main_model:
        model = PreTrainModel(vcblry_size[0], vcblry_size[1])
        paras = list(model.keycode_encoder.parameters()) + list(model.decoder.parameters())
        optimizer = optim.Adam([
            {'params': model.keycode_encoder.parameters(), 'lr': config.learning_rate},
            {'params': model.decoder.parameters(), 'lr': config.learning_rate},
        ], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    else:
        model = MainModel(vcblry_size[0], vcblry_size[1], vcblry_size[2], vcblry_size[3])
        # model.keycode_encoder.load_state_dict(pretrain_kyc_enc, strict=False)
        paras = list(model.keycode_encoder.parameters()) + list(model.sbt_encoder.parameters()) \
                + list(model.cfg_encoder.parameters()) \
                + list(model.hidden_merge_layer.parameters()) + list(model.decoder.parameters())
        optimizer = optim.Adam([
            {'params': model.keycode_encoder.parameters(), 'lr': config.learning_rate},
            {'params': model.sbt_encoder.parameters(), 'lr': config.learning_rate},
            {'params': model.cfg_encoder.parameters(), 'lr': config.learning_rate},
            {'params': model.hidden_merge_layer.parameters(), 'lr': config.learning_rate},
            {'params': model.decoder.parameters(), 'lr': config.learning_rate},
        ], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    lr_schdlr = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_every, gamma=config.lr_decay_rate)

    min_loss = float('inf')
    best_model_state_dict = {}
    early_stopping_rounds = config.main_early_stopping_rounds if is_main_model else config.pretrain_early_stopping_rounds
    round_count = 0

    total_iter = dataset_size // config.batch_size + 1

    # train
    criterion = nn.NLLLoss(ignore_index=nl_vcblry.word2idx['<PAD>'])
    start_time = datetime.now()
    for epoch in range(config.num_epoch):
        print_loss = 0
        last_print_num = 0
        for b_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            decoder_outputs = model(batch, nl_vcblry.word2idx['<BOS>'])
            decoder_outputs = decoder_outputs.view(-1, len(nl_vcblry))

            nl_batch = batch[4].view(-1)
            loss = criterion(decoder_outputs, nl_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(paras, 5)
            optimizer.step()

            print_loss += loss.item()
            if b_idx % config.print_every_batch_num == 0:
                if b_idx - last_print_num != 0:
                    print_loss /= (b_idx - last_print_num)
                logger.info('[Time taken: {!s}], epoch: {}/{}, batch: {}/{}, average loss:{:.4f}'.format(
                    datetime.now() - start_time,
                    epoch, config.num_epoch, b_idx, total_iter, print_loss))
                print_loss = 0
                last_print_num = b_idx

            if not is_main_model:
                if b_idx % config.pretrain_valid_every_iter == 0 and b_idx != 0:
                    logger.info(f'Validting at epoch {epoch}, batch {b_idx}...')
                    cur_state_dict = get_state_dict(model, is_main_model)
                    cur_loss = valid(vcblry_size, valid_loader, cur_state_dict, nl_vcblry, is_main_model)
                    if cur_loss < min_loss:
                        min_loss = cur_loss
                        round_count = 0
                        best_model_state_dict = cur_state_dict
                        best_model_state_dict['epoch'] = epoch
                    else:
                        round_count += 1
                        logger.info(f'Early stop round count: {round_count}/{early_stopping_rounds}.')

                    if round_count >= early_stopping_rounds:
                        logger.info('Early stop!')
                        break
            if round_count >= early_stopping_rounds:
                break

        logger.info(f'Validting at epoch {epoch} end...')
        cur_state_dict = get_state_dict(model, is_main_model)
        cur_loss = valid(vcblry_size, valid_loader, cur_state_dict, nl_vcblry, is_main_model, logger)
        if cur_loss < min_loss:
            min_loss = cur_loss
            round_count = 0
            best_model_state_dict = cur_state_dict
            best_model_state_dict['epoch'] = epoch
        else:
            round_count += 1
            logger.info(f'Early stop round count: {round_count}/{early_stopping_rounds}.')

        if round_count >= early_stopping_rounds:
            logger.info('Early stop!')
            break

        lr_schdlr.step()

    # save best model and return
    if is_main_model:
        model_base_path = config.model_base_path + 'main/'
    else:
        model_base_path = config.model_base_path + 'pre_train/'
    model_save_path = model_base_path + time.strftime("%Y%m%d%H%M") \
                      + f'_best_model_epoch{best_model_state_dict["epoch"]}.pt'
    if not os.path.exists(model_base_path):
        os.makedirs(model_base_path)
    torch.save(best_model_state_dict, model_save_path)

    return best_model_state_dict


def get_state_dict(model, is_main_model):
    model_state_dict = {
        'keycode_encoder': model.keycode_encoder.state_dict(),
        'decoder': model.decoder.state_dict()
    }
    if is_main_model:
        model_state_dict['sbt_encoder'] = model.sbt_encoder.state_dict()
        model_state_dict['cfg_encoder'] = model.cfg_encoder.state_dict()
        model_state_dict['hidden_merge_layer'] = model.hidden_merge_layer.state_dict()
    return model_state_dict


def valid(vcblry_size, valid_loader, cur_state_dict, nl_vcblry, is_main_model, logger):
    if not is_main_model:
        valid_model = PreTrainModel(vcblry_size[0], vcblry_size[1], load_dict=cur_state_dict, is_eval=True)
    else:
        valid_model = MainModel(vcblry_size[0], vcblry_size[1], vcblry_size[2], vcblry_size[3], load_dict=cur_state_dict, is_eval=True)

    validset_loss = 0
    criterion = nn.NLLLoss(ignore_index=nl_vcblry.word2idx['<PAD>'])
    for b_idx, batch in enumerate(valid_loader):
        with torch.no_grad():
            decoder_outputs = valid_model(batch, nl_vcblry.word2idx['<BOS>'])
            decoder_outputs = decoder_outputs.view(-1, len(nl_vcblry))
            nl_batch = batch[4].view(-1)
            batch_loss = criterion(decoder_outputs, nl_batch)
        validset_loss += batch_loss.item()

    validset_loss = validset_loss / len(valid_loader)
    logger.info('Validset loss: {:.4f}'.format(validset_loss))
    return validset_loss


# --------------------------------------------- test ------------------------------------------------------------#
def modeltest(model_state_dict, vcblry_size, test_loader, nl_vcblry, dataset_size, is_main_model, logger):
    if not is_main_model:
        test_model = PreTrainModel(vcblry_size[0], vcblry_size[1], load_dict=model_state_dict, is_eval=True)
    else:
        test_model = MainModel(vcblry_size[0], vcblry_size[1], vcblry_size[2], vcblry_size[3], load_dict=model_state_dict, is_eval=True)

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
                keycode_enc_opt, sbt_enc_opt, cfg_enc_opt, dec_hidden = test_model(batch, nl_vcblry.word2idx['<BOS>'], is_test=True)
                batch_idx_seqs = beam_search_sents(test_model, nl_vcblry, batch_size, keycode_enc_opt, dec_hidden,
                                                   sbt_enc_opt, cfg_enc_opt)

            candidates = trans_idx2word(batch_idx_seqs, nl_vcblry)
            batch_s_bleu = cal_batch_sentence_bleu(nl_batch, candidates)

        total_s_bleu += batch_s_bleu
        total_nl_batch += nl_batch
        total_cands += candidates

    with open(config.dataset_base_path + '/preds_code_cfgsbt_sbt_enc.txt', 'w', encoding='utf-8') as fp, \
            open(config.dataset_base_path + '/ref_code_cfgsbt_sbt_enc.txt', 'w', encoding='utf-8') as fr:
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

# ---------------------------------------------build vcblry---------------------------------------------------------#
def build_vcblry(dataset, vtype):
    vcblry = Vocabulary(vtype)
    for seq in dataset:
        vcblry.add_sequence(seq)
    vcblry.trim(config.max_vcblry_size)
    if not os.path.exists(config.vcblry_base_path):
        os.makedirs(config.vcblry_base_path)
    vcblry.save(config.vcblry_base_path)
    return vcblry


def to_idx_seq(batch, vcblry):
    idx_seqs = []
    for word_seq in batch:
        cur_idx_seq = []
        for word in word_seq:
            if word not in vcblry.word2idx:
                cur_idx_seq.append(vcblry.word2idx['<UNK>'])
            else:
                cur_idx_seq.append(vcblry.word2idx[word])
        cur_idx_seq.append(vcblry.word2idx['<EOS>'])
        idx_seqs.append(cur_idx_seq)
    return idx_seqs


def get_seq_lens(batch):
    seq_lens = []
    for seq in batch:
        seq_lens.append(len(seq))
    return seq_lens


def padding(batch, vcblry):
    padded_batch = list(itertools.zip_longest(*batch, fillvalue=vcblry.word2idx['<PAD>']))
    padded_batch = [list(x) for x in padded_batch]
    return torch.tensor(padded_batch, device=config.device).long()


def my_collate_fn(batch, keycode_vcblry, sbt_vcblry, nl_vcblry, cfg_vcblry, is_raw_nl=False):
    keycode_batch = []
    sbt_batch = []
    cfg_batch = []
    nl_batch = []
    for entry in batch[0]:
        keycode_batch.append(entry[0])
        sbt_batch.append(entry[1])
        cfg_batch.append(entry[3])
        nl_batch.append(entry[2])

    keycode_batch = to_idx_seq(keycode_batch, keycode_vcblry)
    sbt_batch = to_idx_seq(sbt_batch, sbt_vcblry)
    cfg_batch = to_idx_seq(cfg_batch, cfg_vcblry)
    if not is_raw_nl:
        nl_batch = to_idx_seq(nl_batch, nl_vcblry)

    # 先get seq lengths再padding
    keycode_seq_lens = get_seq_lens(keycode_batch)
    sbt_seq_lens = get_seq_lens(sbt_batch)
    cfg_seq_lens = get_seq_lens(cfg_batch)
    nl_seq_lens = get_seq_lens(nl_batch)

    keycode_batch = padding(keycode_batch, keycode_vcblry)
    sbt_batch = padding(sbt_batch, sbt_vcblry)
    cfg_batch = padding(cfg_batch, cfg_vcblry)
    if not is_raw_nl:
        nl_batch = padding(nl_batch, nl_vcblry)

    return keycode_batch, keycode_seq_lens, sbt_batch, sbt_seq_lens, nl_batch, nl_seq_lens, cfg_batch, cfg_seq_lens


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  


def get_logger(path):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def load_data(data_file_path):
    dataset = []
    with open(data_file_path, 'r', encoding='utf-8') as read_file:
        for line in read_file:
            dataset.append(line.strip().split(' '))
    return dataset


def data_truncation(dataset, max_length):
    res = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if len(sample) > max_length:
            sample = sample[:max_length]
        res.append(sample)
    return res


class DatasetObject(Dataset):
    def __init__(self, keycode_file_path, sbt_file_path, nl_file_path, cfg_file_path):
        self.keycode_set = load_data(keycode_file_path)
        self.sbt_set = load_data(sbt_file_path)
        self.cfg_set = load_data(cfg_file_path)
        self.nl_set = load_data(nl_file_path)
        assert len(self.keycode_set) == len(self.sbt_set) == len(self.nl_set)==len(self.cfg_set)

        self.keycode_set = data_truncation(self.keycode_set, config.max_keycode_length)
        self.sbt_set = data_truncation(self.sbt_set, config.max_sbt_length)
        self.cfg_set = data_truncation(self.cfg_set, config.max_cfg_length)
        self.nl_set = data_truncation(self.nl_set, config.max_nl_length)

    def __len__(self):
        return len(self.keycode_set)

    def __getitem__(self, idx):
        return self.keycode_set[idx], self.sbt_set[idx], self.nl_set[idx], self.cfg_set[idx]


def nltk_bleu(hypotheses, references, output_path):
    # output_file = open(output_path, 'w', encoding='utf-8')
    # h_file = open(output_path[0:-4] + "_h_file.txt", 'w', encoding='utf-8')
    # l_file = open(output_path[0:-4] + "_l_file.txt", 'w', encoding='utf-8')
    # m_file = open(output_path[0:-4] + "_m_file.txt", 'w', encoding='utf-8')
    refs = []
    count = 0
    total_score = 0.0
    total_bleu1 = 0.0
    total_bleu2 = 0.0
    total_bleu3 = 0.0
    total_bleu4 = 0.0
    highscore = 0
    lowscore = 0
    midscore = 0
    perfect = 0
    cc = SmoothingFunction()
    i = 1
    for hyp, ref in zip(hypotheses, references):
        s = "y: " + ' '.join(ref) + "\n" + "hyp: " + ' '.join(hyp) + "\n"
        # hyp = hyp.split()
        # ref = ref.split()
        if i == 1:
            print(hyp)
            i += 1
        Cumulate_1_gram = 0
        Cumulate_2_gram = 0
        Cumulate_3_gram = 0
        Cumulate_4_gram = 0
        score = 0.0
        if len(hyp) >= 4:
            try:
                score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method3)
                Cumulate_1_gram = sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=cc.method3)
                Cumulate_2_gram = sentence_bleu([ref], hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=cc.method3)
                Cumulate_3_gram = sentence_bleu([ref], hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=cc.method3)
                Cumulate_4_gram = sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method3)

            except Exception as ex:
                print("==ERROR==\n", ex)
                print("=========")
                exit(0)

        total_score += score
        total_bleu1 += Cumulate_1_gram
        total_bleu2 += Cumulate_2_gram
        total_bleu3 += Cumulate_3_gram
        total_bleu4 += Cumulate_4_gram

        count += 1
        s = s + "bleu-4,mothod3: " + str(score) + "\n"
        s = s + "Cumulate_1_gram: " + str(Cumulate_1_gram) + "\n"
        s = s + "Cumulate_2_gram: " + str(Cumulate_2_gram) + "\n"
        s = s + "Cumulate_3_gram: " + str(Cumulate_3_gram) + "\n"
        s = s + "Cumulate_4_gram: " + str(Cumulate_4_gram) + "\n======================\n\n"
        if score == 1:
            perfect += 1
            highscore += 1
            # h_file.write(s)
        elif score > 0.8:
            highscore += 1
            # h_file.write(s)
        elif score < 0.2:
            lowscore += 1
            # l_file.write(s)
        else:
            midscore += 1
            # m_file.write(s)

    avg_score = total_score / count
    print('avg_score: %.4f' % avg_score)
    # output_file.write('avg_score: %.4f\n' % avg_score)
    avg_bleu1 = total_bleu1 / count
    print('avg_bleu1: %.4f' % avg_bleu1)
    # output_file.write('avg_bleu1: %.4f\n' % avg_bleu1)
    avg_bleu2 = total_bleu2 / count
    print('avg_bleu2: %.4f' % avg_bleu2)
    # output_file.write('avg_bleu2: %.4f\n' % avg_bleu2)
    avg_bleu3 = total_bleu3 / count
    print('avg_bleu3: %.4f' % avg_bleu3)
    # output_file.write('avg_bleu3: %.4f\n' % avg_bleu3)
    avg_bleu4 = total_bleu4 / count
    print('avg_bleu4: %.4f' % avg_bleu4)
    # output_file.write('avg_bleu4: %.4f\n' % avg_bleu4)

    print("hightscore:", highscore)
    print("lowscore:", lowscore)
    print("midscore:", midscore)
    '''
    output_file.write('hightscore num: %d\n' % highscore)
    output_file.write('lowscore num: %d\n' % lowscore)
    output_file.write('midscore num: %d\n' % midscore)
    output_file.write('perfect num: %d\n' % perfect)
    output_file.close()
    h_file.close()
    l_file.close()
    m_file.close()
    '''
    return avg_score

if __name__ == '__main__':
    set_random_seed(config.rand_seed)
    logger = get_logger(config.dataset_base_path + 'log_{}'.format(time.strftime("%Y%m%d")))


    #logger.info('Init trainset...')
    trainset = DatasetObject(config.keycode_trainset_path, config.sbt_trainset_path, config.nl_trainset_path, config.cfg_trainset_path)
    train_loader = DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True,
                              collate_fn=lambda *x: my_collate_fn(x, keycode_vcblry, sbt_vcblry, nl_vcblry, cfg_vcblry))
    # logger.info('Done')


    #logger.info('Init validset and testset...')
    validset = DatasetObject(config.keycode_validset_path, config.sbt_validset_path, config.nl_validset_path, config.cfg_validset_path)
    # validset = TensorDataset(keycode_validset, sbt_validset, nl_validset)
    valid_loader = DataLoader(dataset=validset, batch_size=config.batch_size, shuffle=True,
                              collate_fn=lambda *x: my_collate_fn(x, keycode_vcblry, sbt_vcblry, nl_vcblry, cfg_vcblry))

    testset = DatasetObject(config.keycode_testset_path, config.sbt_testset_path, config.nl_testset_path, config.cfg_testset_path)
    # testset = TensorDataset(keycode_testset, sbt_testset, nl_testset)
    test_loader = DataLoader(dataset=testset, batch_size=config.batch_size, shuffle=False,
                             collate_fn=lambda *x: my_collate_fn(x, keycode_vcblry, sbt_vcblry, nl_vcblry, cfg_vcblry,
                                                                 is_raw_nl=True))
    logger.info('Done.')


    logger.info('Build vocabularies...')
    keycode_vcblry = build_vcblry(trainset.keycode_set, 'keycode')
    sbt_vcblry = build_vcblry(trainset.sbt_set, 'sbt')
    cfg_vcblry = build_vcblry(trainset.cfg_set, 'cfg')
    nl_vcblry = build_vcblry(trainset.nl_set, 'nl')
    logger.info('Done.')

    keycode_vcblry_size = len(keycode_vcblry)
    sbt_vcblry_size = len(sbt_vcblry)
    cfg_vcblry_size = len(cfg_vcblry)
    nl_vcblry_size = len(nl_vcblry)

    # train
    # logger.info('Training the main model...')
    # start_time = datetime.now()
    #
    # main_model_state = train((keycode_vcblry_size, nl_vcblry_size, sbt_vcblry_size, cfg_vcblry_size),
    #         train_loader, valid_loader, nl_vcblry, len(trainset), is_main_model=True, logger=logger)
    # logger.info('Training done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    # test
    logger.info('Testing the main model...')
    main_model_state = torch.load(f'{config.model_base_path}main/202212071411_best_model_epoch5.pt', map_location='cpu')
    start_time = datetime.now()
    modeltest(main_model_state, (keycode_vcblry_size, nl_vcblry_size, sbt_vcblry_size, cfg_vcblry_size),
         test_loader, nl_vcblry, len(testset), is_main_model=True, logger=logger)
    logger.info('Testing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    ref = []
    hyp = []
    with open(config.dataset_base_path + '/preds_code_cfgsbt_sbt_enc.txt', 'r', encoding='utf-8') as f1, \
        open(config.dataset_base_path + '/ref_code_cfgsbt_sbt_enc.txt', 'r', encoding='utf-8') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for i in range(len(lines1)):
            hyp.append(lines1[i].split()[1:-1])
            ref.append(lines2[i].split()[1:-1])

    nltk_bleu(hyp, ref, '111')
