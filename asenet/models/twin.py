"""TWIN 模型。"""
import tensorflow as tf
import tensorflow.keras as keras

from ..base import BaseHashModel
from ..layers import EfficientMHTA


class TWIN(BaseHashModel):
    def __init__(self, features_dict, seq_cate, seq_brand, target_order, **kwargs):
        super(TWIN, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._seq_cate = seq_cate
        self._seq_brand = seq_brand
        self._target_order = target_order
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)

        self.twin_mhta = EfficientMHTA(num_heads=2, key_dim=8)

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
                ragged = tf.strings.split(inputs.get(key), sep="^")
                dense = ragged.to_tensor(default_value="")
                features_recons[key] = dense
            else:
                features_recons[key] = inputs.get(key)
        return features_recons

    @tf.function
    def _preprocess_features(self, formatted_inputs):
        emb_dict = {}

        # 1. 统一提取并编码所有特征
        for key in self._features_dict.keys():
            val = formatted_inputs.get(key)
            if self._features_dict[key].dtype == tf.string:
                x_int = self._hash_dict.get(key)(val)
                x_emb = self._emb_dict.get(key)(x_int)
                emb_dict[key] = x_emb
            else:
                emb_dict[key] = tf.expand_dims(tf.cast(val, tf.float32), axis=-1)

        # 2. 构造目标特征与历史序列特征
        target_emb = tf.keras.layers.concatenate([emb_dict[f] for f in self._target_order], axis=-1)

        seq_cate = emb_dict[self._seq_cate]
        seq_brand = emb_dict[self._seq_brand]
        seq_len = tf.minimum(tf.shape(seq_cate)[1], tf.shape(seq_brand)[1])
        seq_cate = seq_cate[:, :seq_len, :]
        seq_brand = seq_brand[:, :seq_len, :]
        seq_emb = tf.keras.layers.concatenate([seq_cate, seq_brand], axis=-1)

        # 3. 生成 Mask 屏蔽空白词
        mask_str = formatted_inputs.get(self._seq_cate)[:, :seq_len]
        mask = tf.math.not_equal(mask_str, "")

        # 4. TWIN 核心计算
        user_interest = self.twin_mhta(target_emb, seq_emb, mask)

        # 5. 组合其余单值特征
        final_inputs = [user_interest]
        for key in self._features_dict.keys():
            if self._features_dict[key].is_seq == 0:
                final_inputs.append(emb_dict[key])

        processed_input = tf.keras.layers.concatenate(final_inputs, axis=-1)
        return processed_input

    @tf.function
    def call(self, data):
        data = self._format_inputs(data)
        x = self._preprocess_features(data)
        y_pred = self._dense(x)
        y_pred = keras.activations.sigmoid(y_pred)
        return y_pred


def build_model(features_dict, ctx):
    # 原始实现中 TWIN 目标特征的拼接顺序在两个数据集上不同（需严格保留）。
    if ctx.dataset == "industrial":
        target_order = [ctx.target_brand, ctx.target_cate]  # [seller_id, cate_disp_id]
    else:
        target_order = [ctx.target_cate, ctx.target_brand]  # [cate_id, brand]
    return TWIN(features_dict, seq_cate=ctx.seq_cate, seq_brand=ctx.seq_brand, target_order=target_order)
