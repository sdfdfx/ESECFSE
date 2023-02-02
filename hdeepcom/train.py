import config
import torch.optim as optim
from datetime import datetime
import torch
import torch.nn as nn
import os
import time
from models.pretrain_model import PreTrainModel
from models.main_model import MainModel


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
        model = MainModel(vcblry_size[0], vcblry_size[1], vcblry_size[2])
        # model.keycode_encoder.load_state_dict(pretrain_kyc_enc, strict=False)
        paras = list(model.keycode_encoder.parameters()) + list(model.sbt_encoder.parameters()) \
                + list(model.hidden_merge_layer.parameters()) + list(model.decoder.parameters())
        optimizer = optim.Adam([
            {'params': model.keycode_encoder.parameters(), 'lr': config.learning_rate},
            {'params': model.sbt_encoder.parameters(), 'lr': config.learning_rate},
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
            # 梯度裁剪
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
        model_state_dict['hidden_merge_layer'] = model.hidden_merge_layer.state_dict()
    return model_state_dict


def valid(vcblry_size, valid_loader, cur_state_dict, nl_vcblry, is_main_model, logger):
    if not is_main_model:
        valid_model = PreTrainModel(vcblry_size[0], vcblry_size[1], load_dict=cur_state_dict, is_eval=True)
    else:
        valid_model = MainModel(vcblry_size[0], vcblry_size[1], vcblry_size[2], load_dict=cur_state_dict, is_eval=True)

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
