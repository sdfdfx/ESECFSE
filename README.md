# Towards improving the performance of comment generation models by introducing bytecode information

This is the source code and dataset for ourwork. 

# 1.Dataset
First, download our preprocessed dataset from [google driver](https://drive.google.com/file/d/13F5BIOn6wkg82YkuYon1tfXLJRAWsYin/view?usp=share_link) and extract it to the current directory.<br>
The dataset is saved in three folders: train, test and valid. Includes processed comments, source code, CFG of bytecode, SBT (excluding non-terminal node content, applicable to Hybrid-deepcom model and AST-Attengru model) and SBTdeepcom (including non-terminal and leaf node content, applicable to deepcom model)
# 2.Training
We provide specific implementations of Transformer, Hybrid-deepcom, Deepcom and AST-Attengru. Transformer and AST-Attengru refer to the official implementation code (from [https://github.com/wasiahmad/NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum) and [https://bit.ly/2MLSxFg](https://bit.ly/2MLSxFg)) and make changes on this basis.


## 2.1 train Deepcom
Enter the deepcom folder, and change the config.py file.
#### 2.1.1 train Raw model
> 1.open the config.py file and change the keycode_trainset_path、keycode_validset_path and keycode_testset_path to the path of sbtdeepcom dataset. <br>
> 2.run the train_seq2seq.py <br>
Command: ```python train_seq2seq.py``` <br>

#### 2.1.2 train Replace model
> 1.open the config.py file and change the keycode_trainset_path、keycode_validset_path and keycode_testset_path to the path of cfgsbt dataset. <br>
> 2.run the train_seq2seq.py <br>
Command: ```python train_seq2seq.py``` <br>

#### 2.1.3 train Add-encoder model
> 1.open the config.py file and change the keycode_trainset_path、keycode_validset_path and keycode_testset_path to the path of cfgsbt dataset; change sbt_trainset_path、sbt_validset_path and sbt_testset_pathto the path of sbtdeepcom dataset<br>
> 2.run the train_seq2seq_fuse_enc.py <br>
Command: ```python train_seq2seq_fuse_enc.py``` <br>

#### 2.1.4 train Cat-embedding model
> 1.open the config.py file and change the keycode_trainset_path、keycode_validset_path and keycode_testset_path to the path of cfgsbt dataset; change sbt_trainset_path、sbt_validset_path and sbt_testset_pathto the path of sbtdeepcom dataset<br>
> 2.run the train_seq2seq_fuse_emb.py <br>
Command: ```python train_seq2seq_fuse_emb.py``` <br>

## 2.2 train Hybrid-Deepcom
Enter the hdeepcom folder, and change the config.py file
#### 2.2.1 train Raw model
> 1.open the config.py file and change the keycode_trainset_path、keycode_validset_path and keycode_testset_path to the path of code dataset; change sbt_trainset_path、sbt_validset_path and sbt_testset_pathto the path of sbt dataset.<br>
> 2.run the main.py <br>
Command: ```python main.py``` <br>

#### 2.2.2 train Replace model
> 1.open the config.py file and change the sbt_trainset_path、sbt_validset_path and sbt_testset_path to the path of cfgsbt dataset. <br>
> 2.run the train_seq2seq.py <br>
Command: ```python train_seq2seq.py``` <br>

#### 2.2.3 train Add-encoder model
> 1.open the config.py file and change the keycode_trainset_path、keycode_validset_path and keycode_testset_path to the path of code dataset; change sbt_trainset_path、sbt_validset_path and sbt_testset_pathto the path of sbtdeepcom dataset; change cfg_trainset_path、cfg_validset_path and cfg_testset_path to the path of cfgsbt dataset. <br>
> 2.run the train_fuse_enc.pyy <br>
Command: ```python train_fuse_enc.py``` <br>

#### 2.2.4 train Cat-embedding model
> 1.open the config.py file and change the keycode_trainset_path、keycode_validset_path and keycode_testset_path to the path of code dataset; change sbt_trainset_path、sbt_validset_path and sbt_testset_pathto the path of sbtdeepcom dataset; change cfg_trainset_path、cfg_validset_path and cfg_testset_path to the path of cfgsbt dataset. <br> 
> 2.run the train_fuse_emb.py <br>
Command: ```python train_fuse_emb.py``` <br>

## 2.3 train Transformer
Copy the dataset under the 10w folder to the NeuralCodeSum\data\10w\java folder.<br>
Then enter the NeuralCodeSum/main folder, 
### 2.3.1 train Raw model
> run the train.py <br>
Command: ```python train.py``` <br>

### 2.3.2 train Replace model
> run the train.py <br>
Command: ```python train.py --train_src train/cfgsbt_train.txt --dev_src valid/cfgsbt_valid.txt --model_name cfgsbt``` <br>

### 2.3.3 train Add-encoder model
> 1.open the transformer2.py file in NeuralCodeSum\c2nl\models folder, then set the self.use_two_encoder to True and set the self.fuse_emb to False in Transformer class
> 2.enter the NeuralCodeSum\main folder and run the train_two_encoder.py <br>
Command: ```python train_two_encoder.py  --model_name addencoder``` <br>

### 2.3.4 train Cat-embedding model
> 1.open the transformer2.py file in NeuralCodeSum\c2nl\models folder, then set the self.use_two_encoder to False and set the self.fuse_emb to True in Transformer class
> 2.enter the NeuralCodeSum\main folder and run the train_two_encoder.py <br>
Command: ```python train_two_encoder.py --model_name catemb``` <br>


## 2.4 train AST-Attendgru
Enter the funcom folder, 
### 2.4.1 train Raw model
> run the train.py <br>
Command: ```python train.py --input-type code_sbt --create-data True --data data/Raw --outdir data/Raw``` <br>

### 2.4.2 train Replace model
> run the train.py <br>
Command: ```python train.py --input-type code_cfgsbt --create-data True --data data/Replace --outdir data/Replace``` <br>

### 2.4.3 train Add-encoder model
> run the train.py <br>
Command: ```python train_fuse_encoder.py --create-data True``` <br>

### 2.4.4 train Cat-embedding model
> run the train.py <br>
Command: ```python train_fuse_embedding.py --create-data True``` <br>
