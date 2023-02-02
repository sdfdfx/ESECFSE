import config
import logging
from torch.utils.data import TensorDataset, DataLoader
from vocabulary import Vocabulary
from train import train
from test import test
from datetime import datetime
import os
import torch
from dataset_obj import DatasetObject
import itertools
import random
import time
from nltk.translate.bleu_score import *

import nltk

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


def my_collate_fn(batch, keycode_vcblry, sbt_vcblry, nl_vcblry, is_raw_nl=False):
    keycode_batch = []
    sbt_batch = []
    nl_batch = []
    for entry in batch[0]:
        keycode_batch.append(entry[0])
        sbt_batch.append(entry[1])
        nl_batch.append(entry[2])

    keycode_batch = to_idx_seq(keycode_batch, keycode_vcblry)
    sbt_batch = to_idx_seq(sbt_batch, sbt_vcblry)
    if not is_raw_nl:
        nl_batch = to_idx_seq(nl_batch, nl_vcblry)

    # 先get seq lengths再padding
    keycode_seq_lens = get_seq_lens(keycode_batch)
    sbt_seq_lens = get_seq_lens(sbt_batch)
    nl_seq_lens = get_seq_lens(nl_batch)

    keycode_batch = padding(keycode_batch, keycode_vcblry)
    sbt_batch = padding(sbt_batch, sbt_vcblry)
    if not is_raw_nl:
        nl_batch = padding(nl_batch, nl_vcblry)

    return keycode_batch, keycode_seq_lens, sbt_batch, sbt_seq_lens, nl_batch, nl_seq_lens


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
    trainset = DatasetObject(config.keycode_trainset_path, config.sbt_trainset_path, config.nl_trainset_path)
    train_loader = DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True,
                              collate_fn=lambda *x: my_collate_fn(x, keycode_vcblry, sbt_vcblry, nl_vcblry))
    # logger.info('Done')

    #logger.info('Init validset and testset...')
    validset = DatasetObject(config.keycode_validset_path, config.sbt_validset_path, config.nl_validset_path)
    # validset = TensorDataset(keycode_validset, sbt_validset, nl_validset)
    valid_loader = DataLoader(dataset=validset, batch_size=config.batch_size, shuffle=True,
                              collate_fn=lambda *x: my_collate_fn(x, keycode_vcblry, sbt_vcblry, nl_vcblry))

    testset = DatasetObject(config.keycode_testset_path, config.sbt_testset_path, config.nl_testset_path)
    # testset = TensorDataset(keycode_testset, sbt_testset, nl_testset)
    test_loader = DataLoader(dataset=testset, batch_size=config.batch_size, shuffle=False,
                             collate_fn=lambda *x: my_collate_fn(x, keycode_vcblry, sbt_vcblry, nl_vcblry,
                                                                 is_raw_nl=True))
    logger.info('Done.')

    logger.info('Build vocabularies...')
    keycode_vcblry = build_vcblry(trainset.keycode_set, 'keycode')
    sbt_vcblry = build_vcblry(trainset.sbt_set, 'sbt')
    nl_vcblry = build_vcblry(trainset.nl_set, 'nl')
    logger.info('Done.')

    keycode_vcblry_size = len(keycode_vcblry)
    sbt_vcblry_size = len(sbt_vcblry)
    nl_vcblry_size = len(nl_vcblry)
    
    # train
    # # 加载预训练模型的
    # # main_model_state = train((keycode_vcblry_size, nl_vcblry_size, sbt_vcblry_size),
    # #                          train_loader, valid_loader, nl_vcblry, len(trainset),
    # #                          is_main_model=True, pretrain_kyc_enc=pre_train_model_state, logger=logger)
    #
    # # 不加载预训练模型的
    # main_model_state = train((keycode_vcblry_size, nl_vcblry_size, sbt_vcblry_size),
    #         train_loader, valid_loader, nl_vcblry, len(trainset), is_main_model=True, logger=logger)
    # logger.info('Training done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    # test
    logger.info('Testing the main model...')
    main_model_state = torch.load(f'{config.model_base_path}main/202301121450_best_model_epoch4.pt', map_location='cpu')
    start_time = datetime.now()
    test(main_model_state, (keycode_vcblry_size, nl_vcblry_size, sbt_vcblry_size),
         test_loader, nl_vcblry, len(testset), is_main_model=True, logger=logger)
    logger.info('Testing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    ref = []
    hyp = []
    with open(config.dataset_base_path + '/preds_seq_fuse_enc2.txt', 'r', encoding='utf-8') as f1, \
        open(config.dataset_base_path + '/ref_seq_fuse_enc2.txt', 'r', encoding='utf-8') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for i in range(len(lines1)):
            hyp.append(lines1[i].split()[1:-1])
            ref.append(lines2[i].split()[1:-1])

    nltk_bleu(hyp, ref, '111')
