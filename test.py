import argparse

import numpy as np
from tqdm import tqdm

from model import load_model
from sampler import read_data, batch_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--specific_weight', default=None)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--test_steps', default=1000, type=int)

    conf = parser.parse_args()
    model = load_model(conf.model_dir, conf.specific_weight)
    model.summary(line_length=200)

    data, _, params = read_data(conf.input_dir)
    # TODO johnkim max len을 model에서 불러오기
    generator, _ = batch_iter(data, params, batch_size=8, max_len=30, data_type='test')

    NDCG = 0.0
    HT1 = 0.0
    HT5 = 0.0
    HT10 = 0.0
    valid_user = 0
    g = generator()
    for _ in tqdm(range(conf.test_steps)):
        inputs, outputs = next(g)
        predicts = model.predict(inputs)
        outputs = list(map(lambda x: np.squeeze(x, axis=-1), outputs[0]))
        predicts = list(map(lambda x: np.argsort(-x, axis=-1), predicts))
        batch_size, seq_len = inputs[-1].shape
        for i in range(batch_size):
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
    print('NDCG = {}\nHT1 = {}\nHT5 = {}\nHT10 = {}\n'.format(NDCG, HT1, HT5, HT10))


if __name__ == '__main__':
    main()
