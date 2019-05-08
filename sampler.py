import json
import os
import pickle
import random

import numpy as np

from variables import VALUE_PAD, VALUE_MASK, VALUE_UNK, DATA_SEQUENCE


def read_data(data_path):
    with open(os.path.join(data_path, "data.pickle"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(data_path, "dic.pickle"), "rb") as f:
        dic = pickle.load(f)
    with open(os.path.join(data_path, "params.json"), "r") as f:
        params = json.load(f)
    return data, dic, params


def gen_batch_inputs(data,
                     params,
                     max_len=512,
                     mask_rate=0.15,
                     mask_mask_rate=0.8,
                     mask_random_rate=0.1,
                     random_sample_length=True,
                     minimum_len=3):
    size_token = params['input'][0]['size']
    token_name = params['input'][0]['name']

    # token_inputs, category_inputs, flag_inputs, interval_inputs, masked_inputs = [], [], [], [], []
    inputs = [[] for _ in range(len(params['input']) + 1)]

    outputs = []
    for idx, elem in enumerate(data):
        orig_len = len(elem[token_name])
        cur_len = min(max_len, orig_len) - 1
        if random_sample_length and cur_len > minimum_len:
            cur_len = random.randrange(minimum_len, cur_len + 1)
        rem_len = max_len - cur_len

        bidx = random.randrange(0, orig_len - (cur_len + 1) + 1)
        eidx = bidx + cur_len

        for pi, p in enumerate(params['input'][1:]):
            inputs[pi + 1].append([VALUE_PAD] * rem_len + elem[p['name']][bidx:eidx] + [VALUE_PAD])

        token_input = []
        masked_input = []
        output = []
        for token in elem[token_name][bidx:eidx]:
            output.append(token)
            if np.random.random() < mask_rate:
                masked_input.append(1)
                r = np.random.random()
                if r < mask_mask_rate:
                    token_input.append(VALUE_MASK)
                elif r < mask_mask_rate + mask_random_rate:
                    while True:
                        random_token = random.randrange(0, size_token)
                        if random_token is VALUE_PAD or random_token is VALUE_UNK or random_token is VALUE_MASK:
                            pass
                        else:
                            break
                    token_input.append(random_token)
                else:
                    token_input.append(token)
            else:
                masked_input.append(0)
                token_input.append(token)

        inputs[0].append([VALUE_PAD] * rem_len + token_input + [VALUE_MASK])
        # masked_input에서 0은 non-mask 1은 mask
        inputs[-1].append([0] * rem_len + masked_input + [1])
        next_value = elem[token_name][eidx]
        outputs.append([VALUE_PAD] * rem_len + output + [next_value])
    inputs = [np.asarray(x) for x in inputs]
    outputs = [np.asarray(np.expand_dims(x, axis=-1)) for x in [outputs]]
    return inputs, outputs


def batch_iter(raw_data, params, batch_size, max_len, mask_rate=0.0, data_type='train', shuffle=True):
    num_batches_per_epoch = int((len(raw_data) - 1) / batch_size) + 1

    def data_generator():
        data = np.array(raw_data)
        data_size = len(data)
        while True:
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                if data_type == 'train':
                    target_data = [{k: data[k][:-2] for k in DATA_SEQUENCE} for idx, data in
                                   enumerate(shuffled_data[start_index: end_index])]
                    random_sample_length = True
                    mr = mask_rate
                elif data_type == 'valid':
                    target_data = [{k: data[k][:-1] for k in DATA_SEQUENCE} for idx, data in
                                   enumerate(shuffled_data[start_index: end_index])]
                    random_sample_length = True
                    mr = mask_rate
                else:
                    target_data = shuffled_data[start_index: end_index]
                    random_sample_length = False
                    mr = mask_rate

                # 마지막 index를 무조건 next인식으로 사용하기 위해서 max_len 하나를 삭제
                inputs, outputs = gen_batch_inputs(target_data, params, max_len - 1,
                                                   mask_rate=mr,
                                                   random_sample_length=random_sample_length)
                yield inputs, outputs

    return data_generator, num_batches_per_epoch
