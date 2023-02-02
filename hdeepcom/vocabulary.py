import pickle


class Vocabulary(object):
    def __init__(self, vtype):
        self.vtype = vtype
        self.spec_token = ['<BOS>', '<EOS>', '<PAD>', '<UNK>']
        self.word2idx = {}
        self.word2num = {}
        self.idx2word = {}
        self.word_cnt = 0
        self.add_sequence(self.spec_token)

    def add_sequence(self, sequence):
        for word in sequence:
            self.add_word(word)
            self.add_word(word.split('_')[0])

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.word_cnt
            self.word2num[word] = 1
            self.idx2word[self.word_cnt] = word
            self.word_cnt += 1
        else:
            self.word2num[word] += 1

    def clear(self):
        self.word2idx = {}
        self.word2num = {}
        self.idx2word = {}
        self.word_cnt = 0

    def trim(self, max_vcblry_size):
        if self.word_cnt <= max_vcblry_size:
            return
        for st in self.spec_token:
            self.word2num.pop(st)
        word_lst_des = sorted(self.word2num.items(), key=lambda d: d[1], reverse=True)
        top_word_lst = word_lst_des[: max_vcblry_size - len(self.spec_token)]
        top_word_lst = self.spec_token + [word for word, _ in top_word_lst]

        self.clear()
        self.add_sequence(top_word_lst)

    def __len__(self):
        return self.word_cnt

    def save(self, file_path):
        with open(file_path + self.vtype, 'wb') as save_file:
            pickle.dump(self, save_file)


# vcblry_base_path = r'E:\工作\中大\bytecodePro\dingxi\JinBo\vcblry_hdeepcom_pro\keycode'
# with open(vcblry_base_path, 'rb') as f:
#     data = pickle.load(f)
#     print(len(data))
