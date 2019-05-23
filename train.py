import argparse
import json
import math
import os

import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import CSVLogger

from custom_model_checkpoint import CustomModelCheckpoint
from data_generator import DataGenerator
from model import get_model, load_model
from utils import read_data, str2bool
from variables import MODEL_FILE_FORMAT, LAST_MODEL_FILE_FORMAT


def main():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--train_dir', required=True)

    # learning option
    parser.add_argument("--use_horovod", type=str2bool, nargs='?',
                        const=False, help="Activate nice mode.")
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
    use_horovod = conf.use_horovod

    if use_horovod:
        import horovod.keras as hvd
        hvd.init()
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))

    if not use_horovod or hvd.rank() == 0:
        os.makedirs(conf.train_dir, exist_ok=True)

    data, _, params = read_data(conf.input_dir)

    last_state_path = os.path.join(conf.train_dir, LAST_MODEL_FILE_FORMAT)

    if os.path.exists(last_state_path):
        model, initial_epoch = load_model(conf.train_dir)
    else:
        with open(os.path.join(conf.train_dir, 'config.json'), "w") as f:
            json.dump(vars(conf), f, sort_keys=True, indent=4, separators=(',', ': '))

        model = get_model(
            input_params=params['input'],
            head_num=conf.head_num,
            transformer_num=conf.transformer_num,
            embed_dim=conf.embed_dim,
            feed_forward_dim=conf.feed_forward_dim,
            seq_len=conf.max_len,
            pos_num=conf.pos_num,
            dropout_rate=conf.dropout_rate,
            lr=conf.lr,
            use_horovod=use_horovod
        )
        initial_epoch = 0

    model.summary()

    train_generator = DataGenerator(data, params, conf.batch_size, conf.max_len, conf.mask_rate, data_type='train')
    valid_generator = DataGenerator(data, params, conf.batch_size, conf.max_len, conf.mask_rate, data_type='valid')
    train_steps = len(train_generator)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=conf.early_stop_patience),
    ]
    if use_horovod:
        callbacks.extend(
            [
                hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                hvd.callbacks.MetricAverageCallback(),
                hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1)
            ]
        )
        train_steps = int(math.ceil(train_steps / hvd.size()))

    if not use_horovod or hvd.rank() == 0:
        callbacks.extend([
            CSVLogger(os.path.join(conf.train_dir, "history.txt"), append=True),
            keras.callbacks.TensorBoard(log_dir=os.path.join(conf.train_dir, "graph"), histogram_freq=0,
                                        write_graph=True, write_images=True),
            CustomModelCheckpoint(os.path.join(conf.train_dir, MODEL_FILE_FORMAT),
                                  os.path.join(conf.train_dir, 'last.h5'))
        ])

    callbacks.append(keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))

    if not use_horovod or hvd.rank() == 0:
        verbose = 1
    else:
        verbose = 0

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_steps,
        epochs=conf.epochs,
        validation_data=valid_generator,
        validation_steps=conf.validation_steps,
        # use_multiprocessing=True,
        # workers=6,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=verbose
    )


if __name__ == '__main__':
    main()
