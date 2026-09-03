"""DIN 模型。"""
import tensorflow as tf
import tensorflow.keras as keras

from ..base import BaseHashModel
from ..layers import DINAttention


class DIN(BaseHashModel):
    hashable_extra = ("price",)

    def __init__(self, features_dict, seq_len, seq_cate, seq_brand, target_cate, target_brand, **kwargs):
        super(DIN, self).__init__(**kwargs)
        self._features_dict = features_dict
        self.seq_len = seq_len
        self.seq_cate = seq_cate
        self.seq_brand = seq_brand
        self.target_cate = target_cate
        self.target_brand = target_brand
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)

        self.attention_cate = DINAttention(hidden_units=[64, 32], name="attention_cate")
        self.attention_brand = DINAttention(hidden_units=[64, 32], name="attention_brand")

        self._dense = tf.keras.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, name="dense_1", activation="relu"),
            keras.layers.Dense(64, name="dense_2", activation="relu"),
            keras.layers.Dense(1, name="dense_out"),
        ])

    @tf.function
    def call(self, data):
        # 1. 处理序列特征：分割、填充/截断、哈希、嵌入并创建 Mask
        cate_his_ragged = tf.strings.split(tf.reshape(data[self.seq_cate], [-1]), sep="^")
        cate_his_padded_str = cate_his_ragged.to_tensor(default_value="", shape=[None, self.seq_len])
        cate_his_mask = tf.not_equal(cate_his_padded_str, "")
        cate_his_hashed = self._hash_dict[self.seq_cate](cate_his_padded_str)
        cate_his_emb = self._emb_dict[self.seq_cate](cate_his_hashed)

        brand_his_ragged = tf.strings.split(tf.reshape(data[self.seq_brand], [-1]), sep="^")
        brand_his_padded_str = brand_his_ragged.to_tensor(default_value="", shape=[None, self.seq_len])
        brand_his_mask = tf.not_equal(brand_his_padded_str, "")
        brand_his_hashed = self._hash_dict[self.seq_brand](brand_his_padded_str)
        brand_his_emb = self._emb_dict[self.seq_brand](brand_his_hashed)

        # 2. 获取目标广告（query）的 Embedding
        target_cate_emb = self._emb_dict[self.target_cate](self._hash_dict[self.target_cate](data[self.target_cate]))
        target_brand_emb = self._emb_dict[self.target_brand](self._hash_dict[self.target_brand](data[self.target_brand]))

        # 3. 通过注意力计算用户兴趣表示
        user_interest_cate = self.attention_cate([target_cate_emb, cate_his_emb], mask=cate_his_mask)
        user_interest_brand = self.attention_brand([target_brand_emb, brand_his_emb], mask=brand_his_mask)

        # 4. 其它特征
        other_features_emb = []
        for key in self._features_dict.keys():
            if key in [self.seq_cate, self.seq_brand, self.target_cate, self.target_brand]:
                continue

            if self._features_dict[key].dtype == tf.string:
                x_hashed = self._hash_dict.get(key)(data.get(key))
                x_emb = self._emb_dict.get(key)(x_hashed)
                other_features_emb.append(x_emb)
            elif key == "price" and self._features_dict[key].dtype == tf.float32:
                price_scaled = data.get(key) * 100.0
                price_binned = tf.cast(price_scaled, dtype=tf.int32)
                price_str = tf.strings.as_string(price_binned)
                hashed_feature = self._hash_dict.get(key)(price_str)
                embedded_feature = self._emb_dict.get(key)(hashed_feature)
                other_features_emb.append(embedded_feature)
            elif self._features_dict[key].dtype == tf.float32:
                reshaped_feature = tf.reshape(data.get(key), (-1, 1))
                other_features_emb.append(reshaped_feature)

        # 5. 拼接所有特征向量
        concat_input = tf.keras.layers.concatenate([
            target_cate_emb,
            target_brand_emb,
            user_interest_cate,
            user_interest_brand,
        ] + other_features_emb, axis=-1)

        # 6. MLP 预测
        y_pred = self._dense(concat_input)
        y_pred = keras.activations.sigmoid(y_pred)
        return y_pred


def build_model(features_dict, ctx):
    return DIN(
        features_dict,
        seq_len=ctx.max_seq_len,
        seq_cate=ctx.seq_cate,
        seq_brand=ctx.seq_brand,
        target_cate=ctx.target_cate,
        target_brand=ctx.target_brand,
    )
