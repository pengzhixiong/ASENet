"""Naive GAN 基线模型（生成器 + 判别器，未引入 WGAN-GP / MoE）。"""
import tensorflow as tf
import tensorflow.keras as keras

CONTENT_LOSS_WEIGHT = 20.0


class GANFeatureExtractor(keras.Model):
    """特征提取器基类：哈希 + 嵌入，序列特征取平均池化。"""

    def __init__(self, features_dict, **kwargs):
        super(GANFeatureExtractor, self).__init__(**kwargs)
        self.features_dict = {k: v for k, v in features_dict.items() if v.used}
        self.hash_layers = {}
        self.embedding_layers = {}
        for name, spec in self.features_dict.items():
            if spec.dtype == tf.string or name == "price":
                self.hash_layers[name] = keras.layers.Hashing(num_bins=spec.vocab, name=f"{name}_hash")
                self.embedding_layers[name] = keras.layers.Embedding(
                    input_dim=spec.vocab, output_dim=spec.dim, name=f"{name}_embedding")

    def get_embedding_list(self, inputs):
        processed_features = []
        for name, spec in self.features_dict.items():
            feature_tensor = inputs[name]
            if spec.dtype == tf.string:
                if spec.is_seq == 1:
                    split_feature = tf.strings.split(feature_tensor, sep="^")
                    hashed_feature = self.hash_layers[name](split_feature)
                    embedded_feature = self.embedding_layers[name](hashed_feature)
                    pooled_feature = tf.reduce_mean(embedded_feature, axis=1)
                    processed_features.append(pooled_feature)
                else:
                    hashed_feature = self.hash_layers[name](feature_tensor)
                    embedded_feature = self.embedding_layers[name](hashed_feature)
                    processed_features.append(embedded_feature)
            elif name == "price" and spec.dtype == tf.float32:
                price_scaled = feature_tensor * 100.0
                price_binned = tf.cast(price_scaled, dtype=tf.int32)
                price_str = tf.strings.as_string(price_binned)
                hashed_feature = self.hash_layers[name](price_str)
                embedded_feature = self.embedding_layers[name](hashed_feature)
                processed_features.append(embedded_feature)
            elif spec.dtype == tf.float32:
                reshaped_feature = tf.reshape(feature_tensor, (-1, 1))
                processed_features.append(reshaped_feature)
        return processed_features

    def call(self, inputs):
        embedding_list = self.get_embedding_list(inputs)
        return keras.layers.concatenate(embedding_list, axis=-1)


class GANGenerator(GANFeatureExtractor):
    """生成器：标准 CTR 预估模型。"""

    def __init__(self, features_dict, **kwargs):
        super(GANGenerator, self).__init__(features_dict, **kwargs)
        self.mlp = tf.keras.Sequential([
            keras.layers.Dense(128, activation="relu", name="g_dense_1"),
            keras.layers.Dense(64, activation="relu", name="g_dense_2"),
            keras.layers.Dense(1, activation="sigmoid", name="g_output"),
        ], name="GeneratorMLP")

    def call(self, inputs):
        feature_vector = super(GANGenerator, self).call(inputs)
        return self.mlp(feature_vector)


class GANDiscriminator(GANFeatureExtractor):
    """判别器：判断 (特征, CTR) 对的真实性。"""

    def __init__(self, features_dict, **kwargs):
        super(GANDiscriminator, self).__init__(features_dict, **kwargs)
        self.mlp = tf.keras.Sequential([
            keras.layers.Dense(128, activation="relu", name="d_dense_1"),
            keras.layers.Dense(64, activation="relu", name="d_dense_2"),
            keras.layers.Dense(1, activation=None, name="d_output"),
        ], name="DiscriminatorMLP")

    def call(self, inputs):
        features, ctr = inputs
        feature_vector = super(GANDiscriminator, self).call(features)
        ctr_reshaped = tf.reshape(ctr, shape=(-1, 1))

        combined_input = keras.layers.concatenate([feature_vector, ctr_reshaped], axis=-1)
        return self.mlp(combined_input)


class NaiveGAN(keras.Model):
    """Naive GAN 封装：G 与 D 及自定义训练逻辑。"""

    def __init__(self, generator, discriminator, **kwargs):
        super(NaiveGAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.ctr_loss_tracker = keras.metrics.Mean(name="ctr_loss")
        self.adv_loss_tracker = keras.metrics.Mean(name="adv_loss")
        self.auc_tracker = tf.keras.metrics.AUC(name="g_auc")

    def compile(self, optimizer_g, optimizer_d, loss_fn):
        super(NaiveGAN, self).compile()
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker, self.ctr_loss_tracker,
                self.adv_loss_tracker, self.auc_tracker]

    @tf.function
    def train_step(self, data):
        features, real_labels = data
        real_labels = tf.cast(real_labels, tf.float32)
        batch_size = tf.shape(real_labels)[0]

        valid = tf.ones((batch_size, 1)) * 0.9
        fake = tf.ones((batch_size, 1)) * 0.1

        with tf.GradientTape() as tape:
            real_pred = self.discriminator([features, real_labels])
            d_loss_real = self.loss_fn(valid, real_pred)

            gen_ctrs = self.generator(features, training=True)
            fake_pred = self.discriminator([features, gen_ctrs])
            d_loss_fake = self.loss_fn(fake, fake_pred)

            d_loss = (d_loss_real + d_loss_fake) / 2

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.optimizer_d.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            gen_ctrs = self.generator(features, training=True)
            validity = self.discriminator([features, gen_ctrs], training=True)

            loss_ctr = self.loss_fn(real_labels, gen_ctrs)
            loss_adv = self.loss_fn(valid, validity)

            g_loss = CONTENT_LOSS_WEIGHT * loss_ctr + loss_adv

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.optimizer_g.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.ctr_loss_tracker.update_state(loss_ctr)
        self.adv_loss_tracker.update_state(loss_adv)
        self.auc_tracker.update_state(real_labels, gen_ctrs)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        features, real_labels = data
        real_labels = tf.cast(real_labels, tf.float32)
        gen_ctrs = self.generator(features, training=False)
        loss_ctr = self.loss_fn(real_labels, gen_ctrs)
        self.ctr_loss_tracker.update_state(loss_ctr)
        self.auc_tracker.update_state(real_labels, gen_ctrs)
        return {"val_ctr_loss": self.ctr_loss_tracker.result(), "val_g_auc": self.auc_tracker.result()}


def build_gan(features_dict, ctx):
    generator = GANGenerator(features_dict)
    discriminator = GANDiscriminator(features_dict)
    return NaiveGAN(generator=generator, discriminator=discriminator)
