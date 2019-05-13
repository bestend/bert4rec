import argparse
import concurrent.futures
import json
import os
from math import ceil

import numpy as np
import scipy.stats
from tqdm import tqdm

from data_generator import DataGenerator
from model import load_model
from utils import read_data
from variables import VALUE_UNK


def predict(data_list, model_dir, specific_weight, show_summary=True, gpuid=None):
    if gpuid is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

    model, _ = load_model(model_dir, specific_weight)
    if show_summary:
        model.summary(line_length=200)

    NDCG = 0.0
    HT1 = 0.0
    HT5 = 0.0
    HT10 = 0.0
    valid_user = 0
    unknown = 0.0
    if show_summary:
        data_list = tqdm(data_list)
    for data in data_list:
        inputs, outputs = data
        predicts = model.predict(inputs)
        outputs = outputs[0][:, -1, 0]
        ranks = scipy.stats.mstats.rankdata(predicts[:, -1, :] * -1, axis=1)
        batch_size, seq_len = inputs[-1].shape
        for i in range(batch_size):
            if outputs[i] == VALUE_UNK:
                unknown += 1
                continue
            rank = ranks[i][outputs[i]]
            if rank <= 1:
                HT1 += 1
            if rank <= 5:
                HT5 += 1
            if rank <= 10:
                HT10 += 1
                NDCG += 1 / np.log2(rank + 2)

        valid_user += batch_size
    NDCG /= valid_user
    HT1 /= valid_user
    HT5 /= valid_user
    HT10 /= valid_user
    unknown /= valid_user
    return NDCG, HT1, HT5, HT10, unknown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--gpu_num', default=1, type=int)
    parser.add_argument('--specific_weight', default='')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--test_steps', default=10, type=int)

    conf = parser.parse_args()

    data, _, params = read_data(conf.input_dir)
    with open(os.path.join(conf.model_dir, "config.json"), "r") as f:
        old_conf = json.load(f)
        max_len = old_conf['max_len']
    generator = DataGenerator(data, params, int(conf.batch_size / conf.gpu_num), max_len, mask_rate=0.0,
                              data_type='test')
    data_list = [generator[idx] for idx in range(conf.test_steps * conf.gpu_num)]
    if conf.gpu_num == 1:
        NDCG, HT1, HT5, HT10, unknown = predict(data_list, conf.model_dir, conf.specific_weight, show_summary=True,
                                                gpuid=None)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=conf.gpu_num) as executor:
            data_size = len(data_list)
            per_data = int(ceil(data_size / conf.gpu_num))
            futures = []
            for idx in range(conf.gpu_num):
                sub_data_list = data_list[per_data * idx: min(per_data * (idx + 1), data_size)]
                future = executor.submit(predict, sub_data_list, conf.model_dir, conf.specific_weight,
                                         show_summary=idx == 0, gpuid=idx)
                futures.append(future)

            result = np.array([future.result() for future in futures]).sum(axis=0) / conf.gpu_num
            NDCG, HT1, HT5, HT10, unknown = result

    print('NDCG = {}\nHT1 = {}\nHT5 = {}\nHT10 = {}\nUNK = {}'.format(NDCG, HT1, HT5, HT10, unknown))


if __name__ == '__main__':
    main()
