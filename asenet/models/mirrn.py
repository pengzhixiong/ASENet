"""MIRRN 模型。"""
import tensorflow as tf
import tensorflow.keras as keras

from ..base import BaseHashModel
from ..layers import DotTargetAttention, MHFTLayer


class MIRRN(BaseHashModel):
    def __init__(self, features_dict, max_seq_len, local_seq_len,
                 seq_cate, seq_brand, target_cate, target_brand, **kwargs):
        super(MIRRN, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._seq_cate = seq_cate
        self._seq_brand = seq_brand
        self._target_names = [target_cate, target_brand]
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)

        # 三个序列 [全局, 局部, 目标感知] 的长度
        self.seq_lengths = [max_seq_len, local_seq_len, max_seq_len]

        self.tapes = [keras.layers.Embedding(input_dim=l, output_dim=16) for l in self.seq_lengths]
        self.mhft_layers = [MHFTLayer(seq_len=l, emb_dim=16) for l in self.seq_lengths]

        self.target_attention = DotTargetAttention()
        self.miam_attention = keras.layers.MultiHeadAttention(num_heads=1, key_dim=8)
        self._dense = tf.keras.Sequential([
            keras.layers.Dense(128, name="mlp_1", activation="relu"),
            keras.layers.Dense(64, name="mlp_2", activation="relu"),
            keras.layers.Dense(1, name="prediction_layer", activation="sigmoid"),
        ])

    @tf.function
    def _format_inputs(self, inputs):
        features_recons = {}
        for key in self._features_dict.keys():
            if self._features_dict[key].is_seq == 1:
                ragged = tf.strings.split(tf.reshape(inputs.get(key), [-1]), sep="^")
                tensor = ragged.to_tensor(default_value="", shape=[None, self.seq_lengths[0]])
                features_recons[key] = tensor
            else:
                features_recons[key] = tf.reshape(inputs.get(key), [-1])
        return features_recons

    def _get_bsrm_output(self, seq_emb, i):
        """Behavior Sequence Refinement Module"""
        seq_len = self.seq_lengths[i]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_emb = self.tapes[i](positions)  # [S, E]

        # 1. Target-Aware Position Encoding (TAPE)
        seq_emb = seq_emb + tf.expand_dims(pos_emb, axis=0)  # [B, S, E] + [1, S, E]

        # 2. Multi-Head Fourier Transformer (MHFT)
        refined_seq = self.mhft_layers[i](seq_emb)

        # 3. Average Pooling
        return tf.reduce_mean(refined_seq, axis=1)  # [B, E]

    @tf.function
    def call(self, data):
        data = self._format_inputs(data)

        dense_inputs = []
        target_embs = []
        hist_embs_cate, hist_embs_brand = None, None

        for key in self._features_dict.keys():
            if self._features_dict[key].dtype == tf.string:
                x_int = self._hash_dict.get(key)(data.get(key))
                x_emb = self._emb_dict.get(key)(x_int)

                if key in self._target_names:
                    target_embs.append(x_emb)
                elif key == self._seq_cate:
                    hist_embs_cate = x_emb
                elif key == self._seq_brand:
                    hist_embs_brand = x_emb
                elif self._features_dict[key].is_seq != 1:
                    dense_inputs.append(x_emb)
            else:
                dense_inputs.append(tf.expand_dims(data.get(key), -1))

        target_item = tf.concat(target_embs, axis=-1)  # [B, 16]
        hist_seq = tf.concat([hist_embs_cate, hist_embs_brand], axis=-1)  # [B, seq_len, 16]

        # 2. MIRM: 构造三种粒度的子序列
        seq_g = hist_seq
        seq_l = hist_seq[:, :self.seq_lengths[1], :]

        _, target_weights = self.target_attention(target_item, hist_seq)
        seq_t = hist_seq * tf.expand_dims(target_weights, axis=-1)

        # 3. BSRM
        E_g = self._get_bsrm_output(seq_g, 0)
        E_l = self._get_bsrm_output(seq_l, 1)
        E_t = self._get_bsrm_output(seq_t, 2)

        # 4. MIAM
        E_all = tf.stack([E_g, E_l, E_t], axis=1)  # [B, 3, 16]
        target_query = tf.expand_dims(target_item, axis=1)  # [B, 1, 16]

        E_u = self.miam_attention(query=target_query, value=E_all, key=E_all)
        E_u = tf.squeeze(E_u, axis=1)  # [B, 16]

        # 5. Prediction Layer
        context_features = tf.keras.layers.concatenate(dense_inputs, axis=-1)
        final_input = tf.keras.layers.concatenate([target_item, E_u, context_features], axis=-1)

        y_pred = self._dense(final_input)
        return y_pred


def build_model(features_dict, ctx):
    return MIRRN(
        features_dict,
        max_seq_len=ctx.max_seq_len,
        local_seq_len=ctx.local_seq_len,
        seq_cate=ctx.seq_cate,
        seq_brand=ctx.seq_brand,
        target_cate=ctx.target_cate,
        target_brand=ctx.target_brand,
    )
