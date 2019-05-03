import argparse
import os

import keras
from keras.callbacks import CSVLogger

from model import get_model
from sampler import read_data, batch_iter


def main():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--train_dir', required=True)

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

    # data 읽어서 처리하고
    data, _, params = read_data(conf.input_dir)

    model = get_model(
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
    model.summary()
    with open(os.path.join(conf.train_dir, "model.json"), "w") as f:
        f.write(model.to_json(sort_keys=True, indent=4, separators=(',', ': ')))

    train_generator, train_step = batch_iter(data, params, conf.batch_size, conf.max_len, conf.mask_rate,
                                             data_type='train')
    valid_generator, _ = batch_iter(data, params, conf.batch_size, conf.max_len, conf.mask_rate, data_type='valid')

    train_step = int(train_step / conf.gpu_num)

    model.fit_generator(
        generator=train_generator(),
        steps_per_epoch=train_step,
        epochs=conf.epochs,
        validation_data=valid_generator(),
        validation_steps=conf.validation_steps,
        # use_multiprocessing=True,
        # workers=6,
        callbacks=[
            CSVLogger(os.path.join(conf.train_dir, "history.txt"), append=True),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=conf.early_stop_patience),
            keras.callbacks.ModelCheckpoint(os.path.join(conf.train_dir, 'weights{epoch:03d}.h5'),
                                            monitor='val_loss',
                                            save_best_only=True,
                                            save_weights_only=True)

        ],
    )


if __name__ == '__main__':
    main()
