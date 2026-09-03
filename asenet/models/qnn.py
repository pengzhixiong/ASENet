"""QNN-α 模型（含 Self-Ensemble Loss）。"""
import tensorflow as tf
import tensorflow.keras as keras

from ..base import BaseHashModel
from ..layers import QNNAlphaLayer


class QNN(BaseHashModel):
    def __init__(self, features_dict, **kwargs):
        super(QNN, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)

        # 统一特征维度到 128，使其能被 num_heads 整除
        self.project_layer = keras.layers.Dense(128, activation="relu", name="feature_projection")

        self.qnn_alpha = QNNAlphaLayer(num_layers=1, num_heads=4, dropout_rate=0.1)

        self.final_dense = keras.layers.Dense(1, activation="sigmoid", name="dense_out")

    @tf.function
    def _format_inputs(self, inputs):
        features_recons = {}
        for key in self._features_dict.keys():
            if self._features_dict[key].is_seq == 1:
                features_recons[key] = tf.strings.split(tf.reshape(inputs.get(key), [-1]), sep="^")
            else:
                features_recons[key] = tf.reshape(inputs.get(key), [-1, 1])
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
                else:
                    x_emb = tf.reshape(x_emb, [-1, self._features_dict[key].dim])
                processed_inputs.append(x_emb)
            else:
                processed_inputs.append(tf.cast(formatted_inputs.get(key), tf.float32))
        processed_input = tf.keras.layers.concatenate(processed_inputs, axis=-1)
        return processed_input

    @tf.function
    def call(self, data, training=False):
        data = self._format_inputs(data)
        x = self._preprocess_features(data)

        x = self.project_layer(x)
        x = self.qnn_alpha(x, training=training)
        y_pred = self.final_dense(x)
        return y_pred

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred1 = self(x, training=True)
            y_pred2 = self(x, training=True)

            epsilon = 1e-7
            y_pred1 = tf.clip_by_value(y_pred1, epsilon, 1.0 - epsilon)
            y_pred2 = tf.clip_by_value(y_pred2, epsilon, 1.0 - epsilon)

            y_pred_avg = (y_pred1 + y_pred2) / 2.0

            bce_loss = self.compiled_loss(y, y_pred_avg, regularization_losses=self.losses)

            y_tilde = tf.stop_gradient(y_pred_avg)
            se_loss = -tf.reduce_mean(
                y_tilde * tf.math.log(y_pred1 * y_pred2 + epsilon)
                + (1 - y_tilde) * tf.math.log((1 - y_pred1) * (1 - y_pred2) + epsilon)
            )

            total_loss = bce_loss + se_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred_avg)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def build_model(features_dict, ctx):
    return QNN(features_dict)
