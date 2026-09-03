"""DLF 模型。"""
import tensorflow as tf
import tensorflow.keras as keras

from ..base import BaseHashModel
from ..layers import DLFLayer


class DLF(BaseHashModel):
    def __init__(self, features_dict, num_dlf_layers=2, **kwargs):
        super(DLF, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)

        self.num_proj = keras.layers.Dense(8, name="num_proj")

        self.num_used_features = len([k for k, v in features_dict.items() if v.used])
        self.flat_dim = self.num_used_features * 8

        self.dlf_layers = [DLFLayer(dim=self.flat_dim) for _ in range(num_dlf_layers)]

        self.final_dense = keras.layers.Dense(1, name="dense_out")

    @tf.function
    def _format_inputs(self, inputs):
        features_recons = {}
        for key in self._features_dict.keys():
            if self._features_dict[key].is_seq == 1:
                features_recons[key] = tf.strings.split(tf.reshape(inputs.get(key), [-1]), sep="^")
            else:
                features_recons[key] = tf.reshape(inputs.get(key), [-1])
        return features_recons

    @tf.function
    def _preprocess_features(self, formatted_inputs):
        processed_inputs = []
        for key in self._features_dict.keys():
            if self._features_dict[key].dtype == tf.string:
                x_int = self._hash_dict.get(key)(formatted_inputs.get(key))
                x_emb = self._emb_dict.get(key)(x_int)
                if self._features_dict[key].is_seq == 1:
                    x_emb = tf.reduce_mean(x_emb, axis=-2)
                processed_inputs.append(x_emb)
            else:
                num_val = tf.expand_dims(formatted_inputs.get(key), -1)  # [Batch] -> [Batch, 1]
                num_emb = self.num_proj(num_val)  # [Batch, 1] -> [Batch, 8]
                processed_inputs.append(num_emb)

        stacked_input = tf.stack(processed_inputs, axis=1)
        flat_input = tf.keras.layers.Flatten()(stacked_input)
        return flat_input

    @tf.function
    def call(self, data):
        data = self._format_inputs(data)

        E_1 = self._preprocess_features(data)

        E_l = E_1
        for layer in self.dlf_layers:
            E_l = layer([E_1, E_l])

        logits = self.final_dense(E_l)
        y_pred = keras.activations.sigmoid(logits)
        return y_pred


def build_model(features_dict, ctx):
    return DLF(features_dict, num_dlf_layers=4)
