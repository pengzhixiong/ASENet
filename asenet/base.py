"""模型公共基类：封装各模型重复的 Hashing / Embedding 初始化逻辑。

不同模型对这两者的差异仅体现在：
- `hashable_extra`：除 string 特征外额外需要建 hash/embed 表的特征名（如 'price'）。
- `emb_mask_zero`：Embedding 是否启用 mask_zero（'none' | 'all' | 'seq'）。
"""
import tensorflow as tf
import tensorflow.keras as keras


class BaseHashModel(keras.Model):
    hashable_extra = ()
    emb_mask_zero = "none"

    def _init_hash_dict(self, features_dict):
        result = {}
        for name, spec in features_dict.items():
            if spec.dtype == tf.string or name in self.hashable_extra:
                result[name] = keras.layers.Hashing(num_bins=spec.vocab, name=name + "_hash")
        return result

    def _init_emb_dict(self, features_dict):
        result = {}
        for name, spec in features_dict.items():
            if spec.dtype == tf.string or name in self.hashable_extra:
                mask_zero = (self.emb_mask_zero == "all"
                             or (self.emb_mask_zero == "seq" and spec.is_seq == 1))
                result[name] = keras.layers.Embedding(
                    input_dim=spec.vocab, output_dim=spec.dim,
                    name=name + "_embedding", mask_zero=mask_zero)
        return result
