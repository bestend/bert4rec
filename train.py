import argparse
import json
import os

import keras
from keras.callbacks import CSVLogger

from custom_model_checkpoint import CustomModelCheckpoint
from data_generator import DataGenerator
from model import get_model, load_model
from utils import read_data
from variables import MODEL_FILE_FORMAT, LAST_MODEL_FILE_FORMAT


def main():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--specific_weight', default=None)

    # learning option
    parser.add_argument('--gpu_num', default=1, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--validation_steps', default=1000, type=int)
    parser.add_argument('--early_stop_patience', default=2, type=int)

    # layer style
    parser.add_argument('--head_num', default=1, type=int)
    parser.add_argument('--transformer_num', default=1, type=int)
    parser.add_argument('--embed_dim', default=25, type=int)
    parser.add_argument('--feed_forward_dim', default=100, type=int)
    parser.add_argument('--max_len', default=30, type=int)
    parser.add_argument('--pos_num', default=30, type=int)
    parser.add_argument('--dropout_rate', default=0.05, type=float)

    # data noise level
    parser.add_argument('--mask_rate', default=0.15, type=float)

    conf = parser.parse_args()

    os.makedirs(conf.train_dir, exist_ok=True)

    data, _, params = read_data(conf.input_dir)

    if conf.specific_weight:
        if os.path.exists(conf.specific_weight):
            print("specific_weight: {}, file not found".format(conf.specific_weight))
        last_state_path = conf.specific_weight
    else:
        last_state_path = os.path.join(conf.train_dir, LAST_MODEL_FILE_FORMAT)

    if os.path.exists(last_state_path):
        model, core_model, initial_epoch = load_model(conf.train_dir, gpu_num=conf.gpu_num)
    else:
        with open(os.path.join(conf.train_dir, 'config.json'), "w") as f:
            json.dump(vars(conf), f, sort_keys=True, indent=4, separators=(',', ': '))

        model, core_model = get_model(
            input_params=params['input'],
            head_num=conf.head_num,
            transformer_num=conf.transformer_num,
            embed_dim=conf.embed_dim,
            feed_forward_dim=conf.feed_forward_dim,
            seq_len=conf.max_len,
            pos_num=conf.pos_num,
            dropout_rate=conf.dropout_rate,
            gpu_num=conf.gpu_num,
        )
        initial_epoch = 0
    core_model.summary()

    train_generator = DataGenerator(data, params, conf.batch_size, conf.max_len, conf.mask_rate, data_type='train')
    valid_generator = DataGenerator(data, params, conf.batch_size, conf.max_len, conf.mask_rate, data_type='valid')

    keras.utils.Sequence()

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=conf.epochs,
        validation_data=valid_generator,
        validation_steps=conf.validation_steps,
        # use_multiprocessing=True,
        # workers=6,
        initial_epoch=initial_epoch,
        callbacks=[
            CSVLogger(os.path.join(conf.train_dir, "history.txt"), append=True),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=conf.early_stop_patience),
            CustomModelCheckpoint(core_model, os.path.join(conf.train_dir, MODEL_FILE_FORMAT),
                                  os.path.join(conf.train_dir, 'last.h5'))
        ],
    )


if __name__ == '__main__':
    main()
