import argparse
import os
from multiprocessing import Queue, Process

import numpy as np
import scipy.stats
from tqdm import tqdm

from data_generator import DataGenerator
from model import load_model
from utils import read_data
from variables import VALUE_UNK


class Worker(Process):
    def __init__(self, gpuid, queue, result_queue, model_dir, specific_weight):
        Process.__init__(self, name='ModelProcessor')
        self._gpuid = gpuid
        self._queue = queue
        self._result_queue = result_queue
        self._model_dir = model_dir
        self._specific_weight = specific_weight

    def run(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)
        model, _, _ = load_model(self._model_dir, self._specific_weight, test_only=True)
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.put(None)
                break
            result = self.predict(model, *data)
            self._result_queue.put(result)

        self._result_queue.put(None)

    def predict(self, model, inputs, outputs):
        predicts = model.predict(inputs, batch_size=inputs[-1].shape[0])
        outputs = outputs[0][:, -1, 0]
        ranks = scipy.stats.mstats.rankdata(predicts[:, -1, :] * -1, axis=1)
        return outputs, ranks


class Collector(Process):
    def __init__(self, gpu_num, result_queue, valid_user, test_steps):
        Process.__init__(self, name='Collector')
        self._result_queue = result_queue
        self._valid_user = valid_user
        self._test_steps = test_steps
        self._gpu_num = gpu_num

    def run(self):
        NDCG = 0.0
        HT1 = 0.0
        HT5 = 0.0
        HT10 = 0.0
        unknown = 0.0

        none_count = 0
        pbar = tqdm(total=self._test_steps)
        while True:
            result = self._result_queue.get()
            if result is None:
                none_count += 1
                if none_count == self._gpu_num:
                    break
            else:
                outputs, ranks = result
                for i in range(len(outputs)):
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
                pbar.update(1)
        pbar.close()

        NDCG /= self._valid_user
        HT1 /= self._valid_user
        HT5 /= self._valid_user
        HT10 /= self._valid_user
        unknown /= self._valid_user
        print('NDCG = {}\nHT1 = {}\nHT5 = {}\nHT10 = {}\nUNK = {}'.format(NDCG, HT1, HT5, HT10, unknown))


class Scheduler:
    def __init__(self, gpuids, model_dir, specific_weight, valid_user, test_steps):
        self._queue = Queue()
        self._result_queue = Queue()
        self._workers = list()
        self._valid_user = valid_user
        for gpuid in gpuids:
            self._workers.append(Worker(gpuid, self._queue, self._result_queue, model_dir, specific_weight))
        self._collector = Collector(len(gpuids), self._result_queue, valid_user, test_steps)

    def start(self, data_list):
        for data in data_list:
            self._queue.put(data)
        self._queue.put(None)

        for worker in self._workers:
            worker.start()
        self._collector.start()

        for worker in self._workers:
            worker.join()
        self._collector.join()


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
    # max_len = model.input_shape[0][1]
    max_len = 30
    generator = DataGenerator(data, params, conf.batch_size, max_len, mask_rate=0.0, data_type='test')

    valid_user = conf.batch_size * conf.test_steps
    scheduler = Scheduler(list(range(conf.gpu_num)), conf.model_dir, conf.specific_weight, valid_user, conf.test_steps)
    data_list = [generator[idx] for idx in range(conf.test_steps)]
    scheduler.start(data_list)


if __name__ == '__main__':
    main()
