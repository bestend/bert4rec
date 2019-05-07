import argparse
import datetime
import json
import os
import pickle
import time
from collections import defaultdict

from tqdm import tqdm

from variables import DATA_SEQUENCE, VALUE_PAD
from variables import TOKEN_MASK, TOKEN_PAD, TOKEN_UNK, VALUE_UNK, VALUE_MASK


def get_file_line_num(path):
    line_num = 0
    with open(path, 'r') as f:
        for line in f:
            line_num += 1
    return line_num


def get_interval(ts_succ, ts_pred):
    FMT = '%Y-%m-%d %H:%M:%S'

    ts0 = time.mktime(datetime.datetime.strptime(ts_pred, FMT).timetuple())
    ts1 = time.mktime(datetime.datetime.strptime(ts_succ, FMT).timetuple())
    interval = ts1 - ts0

    if interval <= 60:
        du = 0
    elif 60 < interval <= 60 * 3:
        du = 1
    elif 60 * 3 < interval <= 60 * 10:
        du = 2
    elif 60 * 10 < interval <= 60 * 60:
        du = 3
    elif 60 * 60 < interval <= 60 * 60 * 6:
        du = 4
    elif 60 * 60 * 6 < interval <= 60 * 60 * 24:
        du = 5
    else:
        du = 6

    return du


def refine(columns):
    interval_index = DATA_SEQUENCE.index('interval')

    size_col = len(columns[interval_index])
    modified_interval = [get_interval(columns[interval_index][i], columns[interval_index][i + 1]) for i in
                         range(size_col - 1)]

    # TODO johnkim padding값이랑 그냥 0이랑 구분이 안됨, 분리해야할듯
    modified_interval.append(VALUE_PAD)
    columns[interval_index] = modified_interval
    return columns


def data_reader(path, column_size):
    incorrect = 0
    total = 0
    with open(path, 'r') as file:
        for line in file:
            total += 1
            line = line.strip()
            tokens = line.split('\t')
            if len(tokens) != column_size:
                incorrect += 1
                continue
            columns = [token.split(",") for token in tokens[1:]]
            if len(columns[0]) is 0 or len(set(map(len, columns))) != 1:
                incorrect += 1
                continue
            columns = refine(columns)
            yield columns

    print("incorrect ratio = {}".format(incorrect / total * 100.0))


def preprocess(data_path):
    data_dict = dict()
    for k in DATA_SEQUENCE:
        data_dict[k] = {
            TOKEN_PAD: VALUE_PAD,
            TOKEN_UNK: VALUE_UNK,
        }

    # 학습을 위한 추가 token
    data_dict['token'][TOKEN_MASK] = VALUE_MASK

    data_frequency = {k: defaultdict(lambda: 0) for k in DATA_SEQUENCE}

    total_behavior = 0
    max_len = 0
    data = []
    for columns in tqdm(data_reader(data_path, len(DATA_SEQUENCE) + 1), total=get_file_line_num(data_path)):
        if len(columns[0]) < 4:
            continue
        total_behavior += len(columns[0])
        for idx, k in enumerate(DATA_SEQUENCE):
            for e in columns[idx]:
                data_frequency[k][e] += 1
                if e not in data_dict[k]:
                    data_dict[k][e] = len(data_dict[k])
            columns[idx] = [data_dict[k][e] for e in columns[idx]]
            max_len = max(max_len, len(columns[0]))
        data.append({k: v for k, v in zip(DATA_SEQUENCE, columns)})

    stat = "total behavior = {}\n".format(total_behavior)
    stat += "max_len = {}\n".format(max_len)
    for k in DATA_SEQUENCE:
        stat += "total {} num = {}\n".format(k, len(data_frequency[k]))

    data_frequency = {k: dict(v) for k, v in data_frequency.items()}

    params = dict()
    params['input'] = [{'name': k, 'size': len(data_dict[k])} for k in DATA_SEQUENCE]
    return params, data, data_dict, data_frequency, stat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_dir', required=True)

    conf = parser.parse_args()

    os.makedirs(conf.output_dir, exist_ok=True)

    params, data, data_dict, data_frequency, stat = preprocess(conf.input)

    print(stat)
    with open(os.path.join(conf.output_dir, 'statistics.txt'), "w") as f:
        f.write(stat)
    with open(os.path.join(conf.output_dir, 'data.pickle'), "wb") as f:
        pickle.dump(data, f)
    with open(os.path.join(conf.output_dir, 'dic.pickle'), "wb") as f:
        pickle.dump(data_dict, f)
    with open(os.path.join(conf.output_dir, 'frequency.pickle'), "wb") as f:
        pickle.dump(data_frequency, f)
    with open(os.path.join(conf.output_dir, 'params.json'), "w") as f:
        json.dump(params, f, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    main()
