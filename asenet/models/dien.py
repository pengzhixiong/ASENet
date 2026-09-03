"""DIEN 模型。"""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from ..base import BaseHashModel
from ..layers import AUGRU, DIENAttention


class DIEN(BaseHashModel):
    hashable_extra = ("price",)
    emb_mask_zero = "all"

    def __init__(self, features_dict, seq_pairs, max_seq_len, **kwargs):
        super(DIEN, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._seq_pairs = seq_pairs
        self._max_seq_len = max_seq_len

        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)

        seq_emb_dim = 0
        for seq_name in self._seq_pairs.keys():
            seq_emb_dim += features_dict[seq_name].dim

        self.gru_extractor = layers.GRU(seq_emb_dim, return_sequences=True, name="interest_extractor")
        self.attention = DIENAttention(name="attention_layer")
        self.augru = AUGRU(seq_emb_dim, name="interest_evolving")

        self.bn = layers.BatchNormalization()
        self.dense1 = layers.Dense(128, activation="relu", name="dense_1")
        self.dense2 = layers.Dense(64, activation="relu", name="dense_2")
        self.final_dense = layers.Dense(1, activation=None, name="dense_out")

    @tf.function
    def _format_inputs(self, inputs):
        features_recons = {}
        for key in self._features_dict.keys():
            val = inputs.get(key)
            if self._features_dict[key].is_seq == 1:
                split_str = tf.strings.split(val, sep="^")
                dense_str = split_str.to_tensor(default_value="0", shape=[None, self._max_seq_len])
                features_recons[key] = dense_str
            else:
                features_recons[key] = tf.reshape(val, [-1, 1])
        return features_recons

    @tf.function
    def call(self, data):
        data = self._format_inputs(data)

        other_features_emb = []
        seq_emb_list = []
        target_emb_list = []
        mask = None

        for key in self._features_dict.keys():
            feature_conf = self._features_dict[key]
            is_seq = feature_conf.is_seq == 1
            dtype = feature_conf.dtype

            if dtype == tf.string:
                hashed = self._hash_dict[key](data[key])
                emb = self._emb_dict[key](hashed)
            elif dtype == tf.float32 and key == "price":
                price_scaled = data.get(key) * 100.0
                price_binned = tf.cast(price_scaled, dtype=tf.int32)
                price_str = tf.strings.as_string(price_binned)
                hashed_feature = self._hash_dict.get(key)(price_str)
                emb = self._emb_dict.get(key)(hashed_feature)
            else:
                emb = data[key]
                if len(emb.shape) == 2 and not is_seq:
                    emb = tf.expand_dims(emb, -1)

            if is_seq:
                seq_emb_list.append(emb)
                if mask is None:
                    mask = tf.not_equal(hashed, 0)
            elif key in self._seq_pairs.values():
                target_emb_list.append(emb)
            else:
                if len(emb.shape) == 3:
                    emb = tf.reduce_sum(emb, axis=1)
                other_features_emb.append(emb)

        # 依赖特征定义顺序保证 seq_emb_list 与 target_emb_list 对应
        history_seq_emb = tf.concat(seq_emb_list, axis=-1)
        target_item_emb = tf.concat(target_emb_list, axis=-1)

        # 1. 兴趣抽取
        gru_out = self.gru_extractor(history_seq_emb, mask=mask)

        # 2. Attention
        att_scores = self.attention([gru_out, target_item_emb], mask=mask)

        # 3. 兴趣进化 (AUGRU)
        final_interest = self.augru([gru_out, att_scores], mask=mask)

        # 4. MLP
        target_item_emb_sq = tf.squeeze(target_item_emb, axis=1)
        all_feats = [final_interest, target_item_emb_sq] + other_features_emb
        deep_input = tf.concat(all_feats, axis=-1)

        x = self.bn(deep_input)
        x = self.dense1(x)
        x = self.dense2(x)
        logits = self.final_dense(x)
        output = keras.activations.sigmoid(logits)
        return output


def build_model(features_dict, ctx):
    seq_pairs = {ctx.seq_cate: ctx.target_cate, ctx.seq_brand: ctx.target_brand}
    return DIEN(features_dict, seq_pairs, max_seq_len=ctx.max_seq_len)
