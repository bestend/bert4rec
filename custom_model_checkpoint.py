import keras
from keras.callbacks import Callback


class CustomModelCheckpoint(Callback):
    def __init__(self, weight_path, last_path):
        self._weight_saver = keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True)
        self._last_saver = keras.callbacks.ModelCheckpoint(last_path)
        pass

    def on_epoch_end(self, epoch, logs=None):
        self._weight_saver.on_epoch_end(epoch, logs)
        self._last_saver.on_epoch_end(epoch, logs)
