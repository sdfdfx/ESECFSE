from torch.utils.data import Dataset
import config


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
    def __init__(self, keycode_file_path, sbt_file_path, nl_file_path):
        self.keycode_set = load_data(keycode_file_path)
        self.sbt_set = load_data(sbt_file_path)
        self.nl_set = load_data(nl_file_path)
        assert len(self.keycode_set) == len(self.sbt_set) == len(self.nl_set)

        self.keycode_set = data_truncation(self.keycode_set, config.max_keycode_length)
        self.sbt_set = data_truncation(self.sbt_set, config.max_sbt_length)
        self.nl_set = data_truncation(self.nl_set, config.max_nl_length)

    def __len__(self):
        return len(self.keycode_set)

    def __getitem__(self, idx):
        return self.keycode_set[idx], self.sbt_set[idx], self.nl_set[idx]
