"""ASENet 模型（LSTM 时序生成器 + MoE 判别器 + WGAN-GP）。"""
import tensorflow as tf
import tensorflow.keras as keras

from ..layers import MoELayer

CONTENT_LOSS_WEIGHT = 20.0
GP_WEIGHT = 10.0
MoE_BALANCE = 0.2
NUM_EXPERTS = 4
EXPERT_HIDDEN_UNITS = [64, 32]


class ASENetFeatureExtractor(keras.Model):
    """特征提取器基类：共享嵌入 + 单值哈希嵌入 + 序列 LSTM 编码。"""

    def __init__(self, features_dict, sequence_max_len, share_groups,
                 mask_zero_non_shared=True, pad_empty_seq=True, **kwargs):
        super(ASENetFeatureExtractor, self).__init__(**kwargs)
        self.features_dict = {k: v for k, v in features_dict.items() if v.used}
        self.sequence_max_len = sequence_max_len
        self.mask_zero_non_shared = mask_zero_non_shared
        self.pad_empty_seq = pad_empty_seq

        # 由 share_groups 推导 sharing_map（成员 -> 组）与 base_feature_map（组 -> base 特征）
        self.sharing_map = {}
        self.base_feature_map = {}
        for group_name, info in share_groups.items():
            self.base_feature_map[group_name] = info["base"]
            for member in info["members"]:
                self.sharing_map[member] = group_name

        self.shared_hash_layers = {}
        self.shared_embedding_layers = {}
        self.hash_layers = {}
        self.embedding_layers = {}
        self.lstm_layers = {}

        for group_name, base_feature_name in self.base_feature_map.items():
            base = self.features_dict[base_feature_name]
            self.shared_hash_layers[group_name] = keras.layers.Hashing(
                num_bins=base.vocab, name=f"shared_{group_name}_hash")
            self.shared_embedding_layers[group_name] = keras.layers.Embedding(
                input_dim=base.vocab, output_dim=base.dim,
                name=f"shared_{group_name}_embedding", mask_zero=True)

        for name, spec in self.features_dict.items():
            # 原始条件：`name not in sharing_map and dtype==tf.string or name=='price'`
            if (name not in self.sharing_map and spec.dtype == tf.string) or name == "price":
                self.hash_layers[name] = keras.layers.Hashing(num_bins=spec.vocab, name=f"{name}_hash")
                self.embedding_layers[name] = keras.layers.Embedding(
                    input_dim=spec.vocab, output_dim=spec.dim,
                    name=f"{name}_embedding", mask_zero=self.mask_zero_non_shared)

            if spec.is_seq == 1:
                self.lstm_layers[name] = keras.layers.LSTM(
                    units=spec.dim, name=f"{name}_lstm", recurrent_activation="sigmoid")

    def get_embedding_list(self, inputs):
        processed_features = []
        for name, spec in self.features_dict.items():
            if spec.dtype != tf.string:
                continue

            feature_tensor = inputs[name]
            if name in self.sharing_map:
                group_name = self.sharing_map[name]
                hash_layer = self.shared_hash_layers[group_name]
                embedding_layer = self.shared_embedding_layers[group_name]
            else:
                hash_layer = self.hash_layers[name]
                embedding_layer = self.embedding_layers[name]

            if spec.is_seq == 1:
                split_feature = tf.strings.split(feature_tensor, sep="^")
                split_feature = tf.ragged.boolean_mask(split_feature, tf.not_equal(split_feature, ""))
                truncated_feature = split_feature[:, :self.sequence_max_len]

                hashed_feature = hash_layer(truncated_feature)
                dense_hashed_feature = hashed_feature.to_tensor()

                if self.pad_empty_seq:
                    # 补齐空序列，确保 LSTM 至少执行 1 步
                    seq_len = tf.shape(dense_hashed_feature)[1]
                    pad_len = tf.maximum(1 - seq_len, 0)
                    dense_hashed_feature = tf.pad(dense_hashed_feature, [[0, 0], [0, pad_len]])

                embedded_feature = embedding_layer(dense_hashed_feature)
                lstm_output = self.lstm_layers[name](embedded_feature)
                processed_features.append(lstm_output)
            else:
                hashed_feature = hash_layer(feature_tensor)
                embedded_feature = embedding_layer(hashed_feature)
                processed_features.append(embedded_feature)

        return processed_features

    def call(self, inputs):
        embedding_list = self.get_embedding_list(inputs)
        return keras.layers.concatenate(embedding_list, axis=-1)


class Generator(ASENetFeatureExtractor):
    """生成器（TIG）：LSTM 时序编码 + MLP 输出 CTR 概率。"""

    def __init__(self, features_dict, **kwargs):
        super(Generator, self).__init__(features_dict, **kwargs)
        self.mlp = tf.keras.Sequential([
            keras.layers.Dense(128, activation="relu", name="g_dense_1"),
            keras.layers.Dense(64, activation="relu", name="g_dense_2"),
            keras.layers.Dense(1, activation="sigmoid", name="g_output"),
        ], name="GeneratorMLP")

    def call(self, inputs):
        feature_vector = super(Generator, self).call(inputs)
        return self.mlp(feature_vector)


class DiscriminatorWithMoE(ASENetFeatureExtractor):
    """判别器（ARD）：MoE 结构判断 (特征, CTR) 对的真实性。"""

    def __init__(self, features_dict, num_experts, expert_hidden_units, **kwargs):
        super(DiscriminatorWithMoE, self).__init__(features_dict, **kwargs)
        self.moe_layer = MoELayer(
            num_experts=num_experts,
            expert_hidden_units=expert_hidden_units,
            output_dim=1,
            name="Discriminator_MoE_Layer",
        )

    def call(self, inputs):
        features, ctr = inputs
        feature_vector = super(DiscriminatorWithMoE, self).call(features)
        ctr_reshaped = tf.reshape(ctr, shape=(-1, 1))

        combined_input = keras.layers.concatenate([feature_vector, ctr_reshaped], axis=-1)

        logits = self.moe_layer(combined_input)
        return logits


class ASENetGAN(keras.Model):
    """ASENet GAN 封装：WGAN-GP + MoE 负载均衡的自定义训练逻辑。"""

    def __init__(self, generator, discriminator, **kwargs):
        super(ASENetGAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.ctr_loss_tracker = keras.metrics.Mean(name="ctr_loss")
        self.adv_loss_tracker = keras.metrics.Mean(name="adv_loss")
        self.auc_tracker = tf.keras.metrics.AUC(name="g_auc")
        self.gp_tracker = keras.metrics.Mean(name="gp")

    def compile(self, optimizer_g, optimizer_d, loss_fn):
        super(ASENetGAN, self).compile()
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker, self.ctr_loss_tracker,
                self.adv_loss_tracker, self.gp_tracker, self.auc_tracker]

    def gradient_penalty(self, batch_size, real_ctrs, fake_ctrs, features):
        """计算 WGAN-GP 梯度惩罚。"""
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        interpolated_ctrs = real_ctrs + alpha * (fake_ctrs - real_ctrs)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_ctrs)
            pred = self.discriminator([features, interpolated_ctrs], training=True)

        grads = gp_tape.gradient(pred, [interpolated_ctrs])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, data):
        features, real_labels = data
        real_labels = tf.cast(real_labels, tf.float32)
        real_labels = tf.reshape(real_labels, (-1, 1))

        batch_size = tf.shape(real_labels)[0]

        valid = tf.ones((batch_size, 1)) * 0.9
        fake = tf.ones((batch_size, 1)) * 0.1

        # --- 训练判别器 ---
        with tf.GradientTape() as tape:
            gen_ctrs = self.generator(features, training=True)

            real_pred = self.discriminator([features, real_labels], training=True)
            fake_pred = self.discriminator([features, gen_ctrs], training=True)
            d_loss_real = self.loss_fn(valid, real_pred)
            d_loss_fake = self.loss_fn(fake, fake_pred)

            c_wasserstein_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
            gp = self.gradient_penalty(batch_size, real_labels, gen_ctrs, features)
            moe_balance_loss = sum(self.discriminator.losses)
            d_loss = c_wasserstein_loss + GP_WEIGHT * gp + MoE_BALANCE * moe_balance_loss

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.optimizer_d.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # --- 训练生成器 ---
        with tf.GradientTape() as tape:
            gen_ctrs = self.generator(features, training=True)
            validity = self.discriminator([features, gen_ctrs], training=True)

            loss_ctr = self.loss_fn(real_labels, gen_ctrs)
            loss_adv = -tf.reduce_mean(validity)

            g_loss = CONTENT_LOSS_WEIGHT * loss_ctr + loss_adv

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.optimizer_g.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.ctr_loss_tracker.update_state(loss_ctr)
        self.adv_loss_tracker.update_state(loss_adv)
        self.auc_tracker.update_state(real_labels, gen_ctrs)
        self.gp_tracker.update_state(gp)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        features, real_labels = data
        real_labels = tf.cast(real_labels, tf.float32)
        gen_ctrs = self.generator(features, training=False)
        loss_ctr = self.loss_fn(real_labels, gen_ctrs)
        self.ctr_loss_tracker.update_state(loss_ctr)
        self.auc_tracker.update_state(real_labels, gen_ctrs)
        return {"val_ctr_loss": self.ctr_loss_tracker.result(), "val_g_auc": self.auc_tracker.result()}


def _extractor_kwargs(ctx):
    is_taobao = ctx.dataset == "taobao"
    return {
        "sequence_max_len": ctx.max_seq_len,
        "share_groups": _share_groups(ctx.dataset),
        # 原始实现差异：Industrial 非共享嵌入启用 mask_zero，且对空序列做 padding
        "mask_zero_non_shared": not is_taobao,
        "pad_empty_seq": not is_taobao,
    }


def _share_groups(dataset):
    from ..config import ASENET_SHARE_GROUPS
    return ASENET_SHARE_GROUPS[dataset]


def build_gan(features_dict, ctx):
    kwargs = _extractor_kwargs(ctx)
    generator = Generator(features_dict, **kwargs)
    discriminator = DiscriminatorWithMoE(
        features_dict,
        num_experts=NUM_EXPERTS,
        expert_hidden_units=EXPERT_HIDDEN_UNITS,
        **kwargs,
    )
    return ASENetGAN(generator=generator, discriminator=discriminator)
