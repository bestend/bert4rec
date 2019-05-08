import glob
import os
import re

import keras
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras_bert import gelu
from keras_bert.layers import Masked
from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention
from keras_pos_embd import PositionEmbedding
from keras_position_wise_feed_forward import FeedForward
from keras_transformer import get_encoders

from embedding import TokenEmbedding, EmbeddingSimilarity
from embedding import get_embedding, get_inputs
from variables import LAST_MODEL_FILE_FORMAT


def get_model(input_params,
              pos_num=512,
              seq_len=512,
              embed_dim=768,
              transformer_num=12,
              head_num=12,
              feed_forward_dim=3072,
              dropout_rate=0.1,
              attention_activation=None,
              feed_forward_activation=gelu,
              custom_layers=None,
              training=True,
              trainable=None,
              lr=1e-4,
              gpu_num=1):
    """Get BERT model.

    See: https://arxiv.org/pdf/1810.04805.pdf

    :param pos_num: Maximum position.
    :param seq_len: Maximum length of the input sequence or None.
    :param embed_dim: Dimensions of embeddings.
    :param transformer_num: Number of transformers.
    :param head_num: Number of heads in multi-head attention in each transformer.
    :param feed_forward_dim: Dimension of the feed forward layer in each transformer.
    :param dropout_rate: Dropout rate.
    :param attention_activation: Activation for attention layers.
    :param feed_forward_activation: Activation for feed-forward layers.
    :param custom_layers: A function that takes the embedding tensor and returns the tensor after feature extraction.
                          Arguments such as `transformer_num` and `head_num` will be ignored if `custom_layer` is not
                          `None`.
    :param training: The built model will be returned if it is `True`, otherwise the input layers and the last feature
                     extraction layer will be returned.
    :param trainable: Whether the model is trainable.
    :param lr: Learning rate.
    :return: The compiled model.
    """
    if trainable is None:
        trainable = training
    inputs = get_inputs(
        input_params=input_params,
        seq_len=seq_len
    )
    embed_layer, embed_weights = get_embedding(
        inputs,
        input_params=input_params,
        embed_dim=embed_dim,
        pos_num=pos_num,
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    transformed = embed_layer
    if custom_layers is not None:
        kwargs = {}
        if keras.utils.generic_utils.has_arg(custom_layers, 'trainable'):
            kwargs['trainable'] = trainable
        transformed = custom_layers(transformed, **kwargs)
    else:
        transformed = get_encoders(
            encoder_num=transformer_num,
            input_layer=transformed,
            head_num=head_num,
            hidden_dim=feed_forward_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    if not training:
        return inputs[:2], transformed

    def create_model():
        mlm_dense_layer = keras.layers.Dense(
            units=embed_dim,
            activation=feed_forward_activation,
            trainable=trainable,
            name='MLM-Dense',
        )(transformed)
        mlm_norm_layer = LayerNormalization(name='MLM-Norm')(mlm_dense_layer)
        mlm_pred_layer = EmbeddingSimilarity(name='MLM-Sim')([mlm_norm_layer, embed_weights])
        masked_layer = Masked(name='MLM')([mlm_pred_layer, inputs[-1]])
        model = keras.models.Model(inputs=inputs, outputs=masked_layer)
        return model

    if gpu_num > 1:
        with tf.device('/cpu:0'):
            model = create_model()
        parallel_model = multi_gpu_model(model, gpus=gpu_num)
        parallel_model.compile(
            optimizer=keras.optimizers.Adam(lr=lr),
            loss=keras.losses.sparse_categorical_crossentropy,
        )
        return parallel_model, model
    else:
        model = create_model()
        model.compile(
            optimizer=keras.optimizers.Adam(lr=lr),
            loss=keras.losses.sparse_categorical_crossentropy,
        )
        return model, model


def get_custom_objects():
    import tensorflow as tf
    return {
        "TokenEmbedding": TokenEmbedding,
        "PositionEmbedding": PositionEmbedding,
        "FeedForward": FeedForward,
        "LayerNormalization": LayerNormalization,
        "MultiHeadAttention": MultiHeadAttention,
        "EmbeddingSimilarity": EmbeddingSimilarity,
        "Masked": Masked,
        "gelu": gelu,
        "tf": tf  # keras multi gpu model bug fix
    }


def get_last_epoch(model_path):
    model_pattern = re.compile(r'^.*weights\.(\d+)\-\d+\.\d+\.h5$')
    matched = model_pattern.match(model_path)
    if not matched:
        last_model = sorted(glob.glob(os.path.join(os.path.dirname(model_path), 'weights*.h5')))[-1]
        matched = model_pattern.match(last_model)
    if not matched:
        print("warning: coudn`t extract last epoch num")
        return 0
    return int(matched.group(1))


def load_model(train_dir, specific_weight='', gpu_num=1):
    try:
        if specific_weight:
            model_path = specific_weight
        else:
            model_path = os.path.join(train_dir, LAST_MODEL_FILE_FORMAT)

        last_epoch = get_last_epoch(model_path)
        print("load from => {}".format(model_path))
        if gpu_num > 1:
            with tf.device('/cpu:0'):
                model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
            parallel_model = multi_gpu_model(model, gpus=gpu_num)
            parallel_model.compile(
                optimizer=model.optimizer,
                loss=model.loss,
            )
            return parallel_model, model, last_epoch
        else:
            model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        return model, model, last_epoch

    except Exception as e:
        print(str(e))
        print("model file not found")
