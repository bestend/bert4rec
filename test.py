import argparse

import numpy as np
from tqdm import tqdm

from data_generator import DataGenerator
from model import load_model
from utils import read_data
from variables import VALUE_UNK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--specific_weight', default='')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--test_steps', default=10, type=int)

    conf = parser.parse_args()
    model, _, _ = load_model(conf.model_dir, conf.specific_weight)
    model.summary(line_length=200)

    data, _, params = read_data(conf.input_dir)
    # TODO johnkim max len을 model에서 불러오기
    max_len = model.input_shape[0][1]
    generator = DataGenerator(data, params, conf.batch_size, max_len, mask_rate=0.0, data_type='test')

    NDCG = 0.0
    HT1 = 0.0
    HT5 = 0.0
    HT10 = 0.0
    valid_user = 0
    unknown = 0.0
    for idx in tqdm(range(conf.test_steps)):
        inputs, outputs = generator[idx]
        predicts = model.predict(inputs, batch_size=conf.batch_size)
        outputs = list(map(lambda x: np.squeeze(x, axis=-1), outputs[0]))
        predicts = list(map(lambda x: np.argsort(-x, axis=-1), predicts))
        batch_size, seq_len = inputs[-1].shape
        for i in range(batch_size):
            if outputs[i][-1] == VALUE_UNK:
                unknown += 1
                continue
            rank = np.where(predicts[i][-1] == outputs[i][-1])[0][0]
            if rank < 1:
                HT1 += 1
            if rank < 5:
                HT5 += 1
            if rank < 10:
                HT10 += 1
                NDCG += 1 / np.log2(rank + 2)

        valid_user += batch_size

    NDCG /= valid_user
    HT1 /= valid_user
    HT5 /= valid_user
    HT10 /= valid_user
    unknown /= valid_user
    print('NDCG = {}\nHT1 = {}\nHT5 = {}\nHT10 = {}\nUNK = {}'.format(NDCG, HT1, HT5, HT10, unknown))


if __name__ == '__main__':
    main()
