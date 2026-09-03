"""FINAL 模型（含 Cross-block Knowledge Transfer 自蒸馏）。"""
import tensorflow as tf
import tensorflow.keras as keras

from ..base import BaseHashModel
from ..layers import FinalBlock


class FINAL(BaseHashModel):
    def __init__(self, features_dict, hidden_dim=128, num_blocks=2, **kwargs):
        super(FINAL, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)

        self.num_blocks = num_blocks
        self.final_blocks = [
            FinalBlock(hidden_dim=hidden_dim, num_layers=2, num_steps=2, name=f"final_block_{i}")
            for i in range(self.num_blocks)
        ]

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
    def _preprocess_features(self, formatted_inputs):
        processed_inputs = []
        for key in self._features_dict.keys():
            if self._features_dict[key].dtype == tf.string:
                x_int = self._hash_dict.get(key)(formatted_inputs.get(key))
                x_emb = self._emb_dict.get(key)(x_int)
                if self._features_dict[key].is_seq == 1:
                    x_emb = tf.reduce_mean(x_emb, axis=-2)
                processed_inputs.append(x_emb)
            else:
                val = formatted_inputs.get(key)
                val = tf.cast(val, tf.float32)
                val = tf.reshape(val, [-1, 1])
                processed_inputs.append(val)

        processed_input = tf.keras.layers.concatenate(processed_inputs, axis=-1)
        return processed_input

    def call(self, data, training=False):
        data = self._format_inputs(data)
        x = self._preprocess_features(data)

        block_logits = [block(x) for block in self.final_blocks]

        if self.num_blocks > 1:
            concat_logits = tf.concat(block_logits, axis=-1)
            final_logit = tf.reduce_mean(concat_logits, axis=-1, keepdims=True)
        else:
            final_logit = block_logits[0]

        y_pred = keras.activations.sigmoid(final_logit)

        if training and self.num_blocks > 1:
            block_preds = [keras.activations.sigmoid(l) for l in block_logits]
            return y_pred, block_preds

        return y_pred

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            if self.num_blocks > 1:
                y_pred, block_preds = self(x, training=True)
                task_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

                kd_loss = 0.0
                teacher_soft_label = tf.stop_gradient(y_pred)
                for bp in block_preds:
                    kd_loss += tf.reduce_mean(keras.losses.binary_crossentropy(teacher_soft_label, bp))

                total_loss = task_loss + kd_loss
            else:
                y_pred = self(x, training=True)
                total_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def build_model(features_dict, ctx):
    return FINAL(features_dict, hidden_dim=128, num_blocks=2)
