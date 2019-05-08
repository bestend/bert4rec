import random

import keras
import numpy as np

from variables import VALUE_PAD, VALUE_MASK, VALUE_UNK, DATA_SEQUENCE


class DataGenerator(keras.utils.Sequence):

    def __init__(self, raw_data, params, batch_size, max_len, mask_rate=0.0, data_type='train', shuffle=True):
        self.data = np.array(raw_data)
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.params = params
        self.max_len = max_len
        self.data_type = data_type
        if self.data_type == 'train':
            self.mask_rate = mask_rate
            self.random_sample_length = True
        elif self.data_type == 'valid':
            self.mask_rate = mask_rate
            self.random_sample_length = True
        else:
            self.mask_rate = 0.0
            self.random_sample_length = False
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.data_size / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.data_size)
        if self.data_type == 'train':
            target_data = [{k: data[k][:-2] for k in DATA_SEQUENCE} for idx, data in
                           enumerate(self.data[start_index: end_index])]
        elif self.data_type == 'valid':
            target_data = [{k: data[k][:-1] for k in DATA_SEQUENCE} for idx, data in
                           enumerate(self.data[start_index: end_index])]
        else:
            target_data = self.data[start_index: end_index]

        inputs, outputs = self.gen_batch_inputs(target_data, self.params, self.max_len - 1, mask_rate=self.mask_rate,
                                                random_sample_length=self.random_sample_length)
        return inputs, outputs

    def on_epoch_end(self):
        if self.shuffle:
            shuffle_indices = np.random.permutation(np.arange(self.data_size))
            self.data = self.data[shuffle_indices]

    def gen_batch_inputs(self,
                         data,
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
