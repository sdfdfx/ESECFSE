import torch
import os
dataset_base_path = r'../10w/'
keycode_trainset_path = dataset_base_path + 'train/sbtdeepcom_train.txt'
sbt_trainset_path = dataset_base_path + 'train/sbtdeepcom_train.txt'
cfg_trainset_path = dataset_base_path + 'train/cfgsbt_train.txt'
nl_trainset_path = dataset_base_path + 'train/tgt_train.txt'

keycode_validset_path = dataset_base_path + 'valid/sbtdeepcom_valid.txt'
sbt_validset_path = dataset_base_path + 'valid/sbtdeepcom_valid.txt'
cfg_validset_path = dataset_base_path + 'valid/cfgsbt_valid.txt'
nl_validset_path = dataset_base_path + 'valid/tgt_valid.txt'

keycode_testset_path = dataset_base_path + 'test/sbtdeepcom_test.txt'
sbt_testset_path = dataset_base_path + 'test/sbtdeepcom_test.txt'
cfg_testset_path = dataset_base_path + 'test/cfgsbt_test.txt'
nl_testset_path = dataset_base_path + 'test/tgt_test.txt'

max_keycode_length = 384 
max_sbt_length = 384
max_cfg_length = 384
max_nl_length = 30
max_translate_length = 30
batch_size = 1

max_vcblry_size = 50000
vcblry_base_path = r'./vcblry_10w_deepcom_sbtdeepcom/'

encoder_hdn_size = 256
decoder_hdn_size = 256
embedding_dim = 256
use_cuda = torch.cuda.is_available()
cuda_id = 0
device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu")
print(device)

use_teacher_forcing = True
teacher_forcing_ratio = 0.5
learning_rate = 0.001
lr_decay_every = 1
lr_decay_rate = 0.98

model_base_path = r'./trained_model_10w_deepcom_sbtdeepcom/'
load_pre_train_model = False
pretrain_early_stopping_rounds = 3
main_early_stopping_rounds = 3
num_epoch = 20
print_every_batch_num = 200
pretrain_valid_every_iter = 5000

beam_width = 5
rand_seed = 28
