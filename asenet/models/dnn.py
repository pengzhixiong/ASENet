"""DNN 基线模型。"""
import tensorflow as tf
import tensorflow.keras as keras

from ..base import BaseHashModel


class DNN(BaseHashModel):
    # Taobao 数据集额外对 price 建 hash/embed 表（Industrial 无 price 特征，等价于无操作）。
    hashable_extra = ("price",)

    def __init__(self, features_dict, **kwargs):
        super(DNN, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)
        self._dense = tf.keras.Sequential([
            keras.layers.Dense(128, name="dense_1", activation="relu"),
            keras.layers.Dense(64, name="dense_2", activation="relu"),
            keras.layers.Dense(1, name="dense_out"),
        ])

    @tf.function
    def _format_inputs(self, inputs):
        features_recons = {}
        for key in self._features_dict.keys():
            if self._features_dict[key].is_seq == 1:
                # 原始实现此处分隔符为 ','
                features_recons[key] = tf.strings.split(tf.reshape(inputs.get(key), [-1]), sep=",")
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
            elif key == "price" and self._features_dict[key].dtype == tf.float32:
                price_scaled = formatted_inputs.get(key) * 100.0
                price_binned = tf.cast(price_scaled, dtype=tf.int32)
                price_str = tf.strings.as_string(price_binned)
                hashed_feature = self._hash_dict.get(key)(price_str)
                embedded_feature = self._emb_dict.get(key)(hashed_feature)
                processed_inputs.append(embedded_feature)
            elif self._features_dict[key].dtype == tf.float32:
                reshaped_feature = tf.reshape(formatted_inputs.get(key), (-1, 1))
                processed_inputs.append(reshaped_feature)
            else:
                processed_inputs.append(formatted_inputs.get(key))
        processed_input = tf.keras.layers.concatenate(processed_inputs, axis=-1)
        return processed_input

    @tf.function
    def call(self, data):
        data = self._format_inputs(data)
        x = self._preprocess_features(data)
        y_pred = self._dense(x)
        y_pred = keras.activations.sigmoid(y_pred)
        return y_pred


def build_model(features_dict, ctx):
    return DNN(features_dict)
