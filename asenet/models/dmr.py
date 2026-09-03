"""DMR 模型。"""
import tensorflow as tf
import tensorflow.keras as keras

from ..base import BaseHashModel
from ..layers import PositionalEmbedding


class DMR(BaseHashModel):
    hashable_extra = ("price",)
    emb_mask_zero = "seq"

    def __init__(self, features_dict, max_seq_len, seq_cate, seq_brand, target_cate, target_brand, **kwargs):
        super(DMR, self).__init__(**kwargs)

        self._features_dict = features_dict
        self.max_seq_len = max_seq_len

        # 1. 定义特征角色
        self.sequence_features = [seq_cate, seq_brand]
        self.target_item_features = {seq_cate: target_cate, seq_brand: target_brand}
        self.other_sparse_features = [f for f in features_dict.keys()
                                      if f not in self.sequence_features
                                      and f not in self.target_item_features.values()
                                      and features_dict[f].dtype == tf.string]
        self.dense_features = [f for f in features_dict.keys() if features_dict[f].dtype == tf.float32]

        # 2. 初始化 Hashing 和 Embedding 层
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)

        # 3. 为每个序列特征创建位置编码层
        self.cate_pos_emb = PositionalEmbedding(self.max_seq_len, features_dict[seq_cate].dim)
        self.brand_pos_emb = PositionalEmbedding(self.max_seq_len, features_dict[seq_brand].dim)
        self.pos_emb_layers = {seq_cate: self.cate_pos_emb, seq_brand: self.brand_pos_emb}

        # 4. U2I 网络的 Attention 层
        self.cate_attention = keras.layers.Attention(use_scale=True, name="cate_attention")
        self.brand_attention = keras.layers.Attention(use_scale=True, name="brand_attention")
        self.attention_layers = {seq_cate: self.cate_attention, seq_brand: self.brand_attention}

        # 5. Rank 网络 MLP
        self.rank_net = tf.keras.Sequential([
            keras.layers.Dense(128, name="dense_1", activation="relu"),
            keras.layers.Dense(64, name="dense_2", activation="relu"),
            keras.layers.Dense(1, name="dense_out"),
        ])

    @tf.function
    def _format_inputs(self, inputs):
        features_recons = {}
        for key in self._features_dict.keys():
            if self._features_dict[key].is_seq == 1:
                split_seq = tf.strings.split(tf.reshape(inputs.get(key), [-1]), sep="^")
                truncated_seq = split_seq[:, :self.max_seq_len]
                dense_seq = truncated_seq.to_tensor(default_value="")
                features_recons[key] = dense_seq
            else:
                features_recons[key] = tf.reshape(inputs.get(key), [-1])
        return features_recons

    @tf.function
    def call(self, data):
        formatted_inputs = self._format_inputs(data)

        feature_embeddings = {}
        all_sparse_features = self.sequence_features + list(self.target_item_features.values()) + self.other_sparse_features
        for f in all_sparse_features:
            hashed_val = self._hash_dict[f](formatted_inputs[f])
            feature_embeddings[f] = self._emb_dict[f](hashed_val)

        # === Stage 1: U2I Network ===
        attention_outputs = []
        for seq_feat in self.sequence_features:
            target_feat = self.target_item_features[seq_feat]

            query = tf.expand_dims(feature_embeddings[target_feat], axis=1)
            hist_seq_emb = feature_embeddings[seq_feat]

            hist_seq_emb_with_pos = self.pos_emb_layers[seq_feat](hist_seq_emb)

            attention_output = self.attention_layers[seq_feat]([query, hist_seq_emb_with_pos])
            attention_output = tf.squeeze(attention_output, axis=1)
            attention_outputs.append(attention_output)

        # === Stage 2: Rank Network ===
        rank_inputs = []
        rank_inputs.extend(attention_outputs)

        for target_feat in self.target_item_features.values():
            rank_inputs.append(feature_embeddings[target_feat])

        for f in self.other_sparse_features:
            rank_inputs.append(feature_embeddings[f])

        for f in self.dense_features:
            if f == "price":
                price_scaled = formatted_inputs.get(f) * 100.0
                price_binned = tf.cast(price_scaled, dtype=tf.int32)
                price_str = tf.strings.as_string(price_binned)
                hashed_feature = self._hash_dict.get(f)(price_str)
                embedded_feature = self._emb_dict.get(f)(hashed_feature)
                rank_inputs.append(embedded_feature)
            else:
                dense_input = tf.expand_dims(tf.cast(formatted_inputs[f], dtype=tf.float32), axis=1)
                rank_inputs.append(dense_input)

        concatenated_input = tf.keras.layers.concatenate(rank_inputs, axis=-1)

        y_pred = self.rank_net(concatenated_input)
        y_pred = keras.activations.sigmoid(y_pred)
        return y_pred


def build_model(features_dict, ctx):
    return DMR(
        features_dict,
        max_seq_len=ctx.max_seq_len,
        seq_cate=ctx.seq_cate,
        seq_brand=ctx.seq_brand,
        target_cate=ctx.target_cate,
        target_brand=ctx.target_brand,
    )
