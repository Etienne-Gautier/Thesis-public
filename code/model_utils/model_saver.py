import os

import tensorflow as tf


class ModelSaver(tf.keras.callbacks.Callback):

    def __init__(self, path: str):
        super().__init__()
        self.path = path
    
    def on_train_batch_end(self, batch: int, logs=None):
        if(batch % 50 == 0):
            # print("Saving weights for model at batch number " + str(batch))
            self.model.save_weights(self.path)
        

    @staticmethod
    def save_model(model, path: str):
        config = model.get_config()
        weights = model.get_weights()
        os.mkdir("models/" + path)
        with open("models/" + path + "/config.txt") as config_file:
            config_file.write(config)
        with open("models/" + path + "/weights.txt") as weight_file:
            weight_file.write(weights)
    
    @staticmethod
    def load_model(path):
        tf.keras.moldels.Sequential.from_config()
