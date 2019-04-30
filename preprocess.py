import argparse
import json
import os
import pickle

from tqdm import tqdm

from variables import TOKEN_MASK, TOKEN_PAD, TOKEN_UNK

# pickles data index
DEAL = 1
CATEGORY = 2
FLAG = 3
INTERVAL = 4


def make_data(raw_data, dic_deal, dic_cat, dic_flag, dic_interval, max_len):
    size_deal = len(dic_deal)
    size_category = len(dic_cat)
    size_flag = len(dic_flag)
    size_interval = len(dic_interval)

    data = []

    incorrect_count = 0
    for idx, elem in enumerate(tqdm(raw_data)):
        cur_len = len(elem[DEAL])
        if cur_len < 4 or \
                cur_len > max_len or \
                cur_len != len(elem[CATEGORY]) or \
                cur_len != len(elem[FLAG]) or \
                cur_len != len(elem[INTERVAL]) or \
                max(elem[DEAL]) > size_deal or \
                max(elem[CATEGORY]) > size_category or \
                max(elem[FLAG]) > size_flag or \
                max(elem[INTERVAL]) > size_interval:
            # print("ID: ({}) data incorrect".format(idx))
            incorrect_count += 1
            continue

        current = dict()

        current['deal'] = elem[DEAL]
        current['category'] = elem[CATEGORY]
        current['flag'] = elem[FLAG]
        current['interval'] = elem[INTERVAL]
        data.append(current)
    print("incorrect count = {}".format(incorrect_count))
    return data


def convert(path):
    with open("%s/dic.pickle" % path, 'rb') as handle:
        dic_deal = pickle.load(handle)
    with open("%s/dic_cat.pickle" % path, 'rb') as handle:
        dic_cat = pickle.load(handle)
    with open("%s/dic_flag.pickle" % path, 'rb') as handle:
        dic_flag = pickle.load(handle)
    with open("%s/dic_interval.pickle" % path, 'rb') as handle:
        dic_interval = pickle.load(handle)
    with open("%s/data_idx_list_train.pickle" % path, 'rb') as handle:
        data_idx_list_train = pickle.load(handle)
    with open("%s/data_idx_list_dev.pickle" % path, 'rb') as handle:
        data_idx_list_dev = pickle.load(handle)
    with open("%s/max_len.pickle" % path, 'rb') as handle:
        max_len = pickle.load(handle)

    dic_deal[TOKEN_MASK] = len(dic_deal)

    params = dict()
    params['value_pad'] = dic_deal[TOKEN_PAD]
    params['value_unk'] = dic_deal[TOKEN_UNK]
    params['value_mask'] = dic_deal[TOKEN_MASK]
    params['input'] = [
        {'name': 'deal', 'size': len(dic_deal)},
        {'name': 'category', 'size': len(dic_cat)},
        {'name': 'flag', 'size': len(dic_flag)},
        {'name': 'interval', 'size': len(dic_interval)}
    ]
    train_data = make_data(data_idx_list_train, dic_deal, dic_cat, dic_flag, dic_interval, max_len)
    dev_data = make_data(data_idx_list_dev, dic_deal, dic_cat, dic_flag, dic_interval, max_len)

    train_data.extend(dev_data)

    return train_data, dic_deal, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)

    conf = parser.parse_args()

    os.makedirs(conf.output_dir, exist_ok=True)

    data, dic, params = convert(conf.input_dir)

    with open(os.path.join(conf.output_dir, 'data.pickle'), "wb") as f:
        pickle.dump(data, f)

    with open(os.path.join(conf.output_dir, 'dic.pickle'), "wb") as f:
        pickle.dump(dic, f)

    with open(os.path.join(conf.output_dir, 'params.json'), "w") as f:
        json.dump(params, f, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    main()
