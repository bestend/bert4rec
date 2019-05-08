import keras
from keras.callbacks import Callback


class CustomModelCheckpoint(Callback):
    def __init__(self, model, weight_path, last_path):
        self._model = model
        self._weight_saver = keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True)
        self._last_saver = keras.callbacks.ModelCheckpoint(last_path)
        self._weight_saver.set_model(model)
        self._last_saver.set_model(model)
        pass

    def set_model(self, model):
        self._original_model = model

    def on_epoch_end(self, epoch, logs=None):
        self._model.compile(loss=self._original_model.loss, optimizer=self._original_model.optimizer)
        self._weight_saver.on_epoch_end(epoch, logs)
        self._last_saver.on_epoch_end(epoch, logs)
