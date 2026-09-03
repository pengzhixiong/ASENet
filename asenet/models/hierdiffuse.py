"""HierDiffuse 模型。"""
import tensorflow as tf
import tensorflow.keras as keras

from ..base import BaseHashModel
from ..layers import HierDiffuseFusion, MLPTargetAttention


class HierDiffuse(BaseHashModel):
    def __init__(self, features_dict, seq_max_len, short_seq_len,
                 seq_cate, seq_brand, target_cate, target_brand, **kwargs):
        super(HierDiffuse, self).__init__(**kwargs)
        self._features_dict = features_dict
        self.seq_max_len = seq_max_len
        self.short_seq_len = short_seq_len
        self._seq_target_pairs = [(seq_cate, target_cate), (seq_brand, target_brand)]
        self._target_names = [target_cate, target_brand]

        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)

        self.long_attention = MLPTargetAttention()
        self.hier_diffuse_fusion = HierDiffuseFusion(dim=8)

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
                ragged_seq = tf.strings.split(tf.reshape(inputs.get(key), [-1]), sep="^")
                dense_seq = ragged_seq.to_tensor(default_value="", shape=[None, self.seq_max_len])
                features_recons[key] = dense_seq
            else:
                features_recons[key] = tf.reshape(inputs.get(key), [-1])
        return features_recons

    @tf.function
    def _preprocess_features(self, formatted_inputs):
        other_features = []
        seq_features_emb = {}
        target_features_emb = {}

        for key in self._features_dict.keys():
            if self._features_dict[key].dtype == tf.string:
                x_int = self._hash_dict.get(key)(formatted_inputs.get(key))
                x_emb = self._emb_dict.get(key)(x_int)

                if self._features_dict[key].is_seq == 1:
                    seq_features_emb[key] = x_emb  # [B, MAX_LEN, D]
                else:
                    other_features.append(x_emb)  # [B, D]
                    if key in self._target_names:
                        target_features_emb[key] = x_emb
            else:
                other_features.append(tf.expand_dims(formatted_inputs.get(key), axis=-1))

        # HierDiffuse 长短期兴趣切分与融合
        fusion_reps = []

        for seq_key, target_key in self._seq_target_pairs:
            if seq_key in seq_features_emb and target_key in target_features_emb:
                full_seq_emb = seq_features_emb[seq_key]  # [B, seq_max_len, D]
                target_emb = target_features_emb[target_key]  # [B, D]

                # 1. 物理切分：长短期序列
                long_seq_emb = full_seq_emb[:, :-self.short_seq_len, :]
                short_seq_emb = full_seq_emb[:, -self.short_seq_len:, :]

                long_mask = tf.not_equal(tf.reduce_sum(tf.abs(long_seq_emb), axis=-1), 0.0)

                # 2. 长期特征 h_L：Target Attention
                h_long = self.long_attention(target_emb, long_seq_emb, mask=long_mask)

                # 3. 核心融合：HierDiffuse 的 TCC 和 SGD
                h_fusion = self.hier_diffuse_fusion(h_long, short_seq_emb, target_emb)
                fusion_reps.append(h_fusion)

        processed_inputs = other_features + fusion_reps
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
    return HierDiffuse(
        features_dict,
        seq_max_len=ctx.max_seq_len,
        short_seq_len=ctx.short_seq_len,
        seq_cate=ctx.seq_cate,
        seq_brand=ctx.seq_brand,
        target_cate=ctx.target_cate,
        target_brand=ctx.target_brand,
    )
