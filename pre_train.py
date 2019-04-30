import argparse
import os

import keras

from model import get_model
from sampler import read_data, batch_iter


def main():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--train_dir', required=True)

    # learning option
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--validation_steps', default=1000, type=int)
    parser.add_argument('--early_stop_patience', default=5, type=int)

    # layer style
    parser.add_argument('--head_num', default=1, type=int)
    parser.add_argument('--transformer_num', default=1, type=int)
    parser.add_argument('--embed_dim', default=25, type=int)
    parser.add_argument('--feed_forward_dim', default=100, type=int)
    parser.add_argument('--max_len', default=30, type=int)
    parser.add_argument('--pos_num', default=30, type=int)
    parser.add_argument('--dropout_rate', default=0.05, type=float)

    # data noise level
    parser.add_argument('--mask_rate', default=0.3, type=float)
    parser.add_argument('--swap_sentence_rate', default=1.0, type=float)

    conf = parser.parse_args()

    os.makedirs(conf.train_dir, exist_ok=True)

    # data 읽어서 처리하고
    data, _, params = read_data(conf.input_dir)

    model = get_model(
        token_num=params['size_deal'],
        category_num=params['size_category'],
        flag_num=params['size_flag'],
        interval_num=params['size_interval'],
        head_num=conf.head_num,
        transformer_num=conf.transformer_num,
        embed_dim=conf.embed_dim,
        feed_forward_dim=conf.feed_forward_dim,
        seq_len=conf.max_len,
        pos_num=conf.pos_num,
        dropout_rate=conf.dropout_rate,
    )
    model.summary()

    train_generator, train_step = batch_iter(data, params, conf.batch_size, conf.max_len, data_type='train')
    valid_generator, _ = batch_iter(data, params, conf.batch_size, conf.max_len, data_type='valid')

    model.fit_generator(
        generator=train_generator(),
        steps_per_epoch=train_step,
        epochs=conf.epochs,
        validation_data=valid_generator(),
        validation_steps=conf.validation_steps,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=conf.early_stop_patience)
        ],
    )

    # model save
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    main()
