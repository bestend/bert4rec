import argparse
import concurrent.futures
import json
import os
from math import ceil

import numpy as np
from keras import Model
from keras.layers import Lambda
from tqdm import tqdm

from data_generator import DataGenerator
from model import load_model, get_best_model_path
from utils import read_data
from variables import VALUE_UNK


def add_rank_layer_to_model(model, k):
    import tensorflow as tf

    def top_k(input, k):
        return tf.nn.top_k(input, k=k).indices

    output = model.outputs[0]
    output = Lambda(lambda x: x[:, -1, :])(output)
    output = Lambda(top_k, arguments={'k': k})(output)
    model = Model(inputs=model.inputs, outputs=output)
    return model


def predict(data_list, model_dir, total_size, show_summary=True, gpuid=None):
    if gpuid is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

    model, _ = load_model(model_dir)
    if show_summary:
        model.summary(line_length=200)

    model = add_rank_layer_to_model(model, 10)

    NDCG = 0.0
    HT1 = 0.0
    HT5 = 0.0
    HT10 = 0.0
    valid_user = 0
    unknown = 0.0

    data_size = len(data_list)
    if show_summary:
        data_list = tqdm(data_list)
    for data in data_list:
        inputs, outputs = data
        batch_size, seq_len = inputs[-1].shape
        outputs = outputs[0][:, -1, 0]
        predicts = model.predict(inputs, batch_size=batch_size)

        for i in range(batch_size):
            find_result = np.where(predicts[i] == outputs[i])
            if outputs[i] == VALUE_UNK:
                unknown += 1
            #    continue
            if find_result[0].size == 0:
                continue
            rank = find_result[0][0]
            if rank < 1:
                HT1 += 1
            if rank < 5:
                HT5 += 1
            if rank < 10:
                HT10 += 1
                NDCG += 1 / np.log2(rank + 2)

        valid_user += batch_size
    NDCG /= valid_user * data_size / total_size
    HT1 /= valid_user * data_size / total_size
    HT5 /= valid_user * data_size / total_size
    HT10 /= valid_user * data_size / total_size
    unknown /= valid_user * data_size / total_size

    return NDCG, HT1, HT5, HT10, unknown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--gpu_num', default=1, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--test_steps', default=0, type=int)

    conf = parser.parse_args()

    data, _, params = read_data(conf.input_dir)
    if os.path.isfile(conf.model_dir):
        model_path = conf.model_dir
    else:
        model_path = get_best_model_path(conf.model_dir)
    with open(os.path.join(os.path.dirname(model_path), "config.json"), "r") as f:
        old_conf = json.load(f)
        max_len = old_conf['max_len']
    generator = DataGenerator(data, params, int(conf.batch_size / conf.gpu_num), max_len, mask_rate=0.0,
                              data_type='test')
    test_steps = len(generator) if conf.test_steps == 0 else conf.test_steps
    test_steps = min(test_steps, len(generator))

    data_list = [generator[idx] for idx in range(test_steps)]
    data_size = len(data_list)
    if conf.gpu_num == 1:
        NDCG, HT1, HT5, HT10, unknown = predict(data_list, model_path, total_size=data_size, show_summary=True,
                                                gpuid=None)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=conf.gpu_num) as executor:
            per_data = int(ceil(data_size / conf.gpu_num))
            futures = []
            for idx in range(conf.gpu_num):
                sub_data_list = data_list[per_data * idx: min(per_data * (idx + 1), data_size)]
                future = executor.submit(predict, sub_data_list, model_path, total_size=data_size,
                                         show_summary=idx == 0, gpuid=idx)
                futures.append(future)

            result = np.array([future.result() for future in futures]).sum(axis=0)
            NDCG, HT1, HT5, HT10, unknown = result

    print('NDCG = {}\nHT1 = {}\nHT5 = {}\nHT10 = {}\nUNK = {}'.format(NDCG, HT1, HT5, HT10, unknown))


if __name__ == '__main__':
    main()
