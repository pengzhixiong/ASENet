"""DeepFM 基线模型。"""
import tensorflow as tf
import tensorflow.keras as keras

from ..base import BaseHashModel


class DeepFM(BaseHashModel):
    hashable_extra = ("price",)

    def __init__(self, features_dict, **kwargs):
        super(DeepFM, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)
        self._linear_emb_dict = self._init_linear_emb_dict(features_dict)

        # 数值特征的线性权重 w
        self._dense_linear_layer = keras.layers.Dense(1, activation=None, use_bias=False, name="dense_linear")

        self._deep_net = tf.keras.Sequential([
            keras.layers.Dense(128, name="deep_dense_1", activation="relu"),
            keras.layers.Dense(64, name="deep_dense_2", activation="relu"),
            keras.layers.Dense(1, name="deep_out"),
        ])

    def _init_linear_emb_dict(self, features_dict):
        # FM 一阶 Embedding (Dim=1)，相当于权重 w
        result = {}
        for name, spec in features_dict.items():
            if spec.dtype == tf.string or name == "price":
                result[name] = keras.layers.Embedding(
                    input_dim=spec.vocab, output_dim=1, name=name + "_linear_embedding")
        return result

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
    def call(self, data):
        data = self._format_inputs(data)

        sparse_embeds = []   # (Batch, k=8)，用于 FM 二阶与 Deep
        linear_terms = []    # (Batch, 1)，用于 FM 一阶
        dense_inputs = []    # 数值特征，直接用于 Deep

        for key, spec in self._features_dict.items():
            input_val = data.get(key)

            if spec.dtype == tf.string:
                hashed_val = self._hash_dict[key](input_val)

                linear_emb = self._linear_emb_dict[key](hashed_val)
                if spec.is_seq == 1:
                    linear_emb = tf.reduce_mean(linear_emb, axis=-2)
                linear_terms.append(linear_emb)

                emb = self._emb_dict[key](hashed_val)
                if spec.is_seq == 1:
                    emb = tf.reduce_mean(emb, axis=-2)
                sparse_embeds.append(emb)
            elif key == "price" and spec.dtype == tf.float32:
                price_scaled = data.get(key) * 100.0
                price_binned = tf.cast(price_scaled, dtype=tf.int32)
                price_str = tf.strings.as_string(price_binned)
                hashed_val = self._hash_dict.get(key)(price_str)
                linear_emb = self._linear_emb_dict[key](hashed_val)
                linear_terms.append(linear_emb)
                emb = self._emb_dict[key](hashed_val)
                sparse_embeds.append(emb)
            else:
                num_val = tf.reshape(input_val, [-1, 1])
                linear_terms.append(self._dense_linear_layer(num_val))
                dense_inputs.append(num_val)

        # 1. FM 一阶
        linear_logit = tf.add_n(linear_terms)

        # 2. FM 二阶
        if len(sparse_embeds) > 0:
            fm_input_emb = tf.stack(sparse_embeds, axis=1)
            sum_square = tf.square(tf.reduce_sum(fm_input_emb, axis=1))
            square_sum = tf.reduce_sum(tf.square(fm_input_emb), axis=1)
            fm_second_order = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1, keepdims=True)
        else:
            fm_second_order = 0.0

        # 3. Deep 部分
        deep_emb_flatten = (tf.keras.layers.Flatten()(tf.stack(sparse_embeds, axis=1))
                            if len(sparse_embeds) > 0 else None)

        if deep_emb_flatten is not None and len(dense_inputs) > 0:
            deep_input = tf.concat([deep_emb_flatten] + dense_inputs, axis=-1)
        elif deep_emb_flatten is not None:
            deep_input = deep_emb_flatten
        else:
            deep_input = tf.concat(dense_inputs, axis=-1)

        deep_logit = self._deep_net(deep_input)

        total_logit = linear_logit + fm_second_order + deep_logit
        y_pred = tf.nn.sigmoid(total_logit)
        return y_pred


def build_model(features_dict, ctx):
    return DeepFM(features_dict)
