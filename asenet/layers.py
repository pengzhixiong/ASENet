"""各模型复用的自定义 Keras 层，代码与原始脚本逐行一致。

按模型命名避免同名冲突（原脚本中多处定义了同名的 AttentionLayer / TargetAttention）。
"""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


# --------------------------------------------------------------------------- #
# DIN 的注意力激活单元（返回加权池化后的兴趣向量）
# --------------------------------------------------------------------------- #
class DINAttention(keras.layers.Layer):
    def __init__(self, hidden_units, activation="relu", **kwargs):
        super(DINAttention, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.dense_layers = [keras.layers.Dense(u, activation=self.activation) for u in self.hidden_units]
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs, mask=None):
        query, keys = inputs
        seq_len = tf.shape(keys)[1]
        query_tiled = tf.tile(tf.expand_dims(query, 1), [1, seq_len, 1])
        concat_features = tf.concat([query_tiled, keys, query_tiled - keys, query_tiled * keys], axis=-1)

        x = concat_features
        for layer in self.dense_layers:
            x = layer(x)
        attention_scores = self.output_layer(x)
        attention_scores = tf.squeeze(attention_scores, axis=-1)

        if mask is not None:
            paddings = tf.ones_like(attention_scores) * (-2 ** 32 + 1)
            attention_scores = tf.where(mask, attention_scores, paddings)

        attention_weights = tf.nn.softmax(attention_scores)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)

        output = tf.reduce_sum(keys * attention_weights, axis=1)
        return output


# --------------------------------------------------------------------------- #
# DIEN 的注意力层（返回 softmax 分数，供 AUGRU 使用）
# --------------------------------------------------------------------------- #
class DIENAttention(layers.Layer):
    def __init__(self, hidden_units=(32, 16), activation="sigmoid", **kwargs):
        super(DIENAttention, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.dense_layers = []

    def build(self, input_shape):
        for unit in self.hidden_units:
            self.dense_layers.append(layers.Dense(unit, activation=self.activation))
        self.out_dense = layers.Dense(1, activation=None)
        self.built = True

    def call(self, inputs, mask=None):
        keys, query = inputs  # keys: History, query: Target

        seq_len = tf.shape(keys)[1]
        query = tf.tile(query, [1, seq_len, 1])

        info = tf.concat([query, keys, query - keys, query * keys], axis=-1)

        for dense in self.dense_layers:
            info = dense(info)

        outputs = self.out_dense(info)

        if mask is not None:
            padding_mask = tf.cast(mask, tf.float32)
            padding_mask = tf.expand_dims(padding_mask, -1)
            outputs = outputs * padding_mask + (1 - padding_mask) * (-1e9)

        scores = tf.nn.softmax(outputs, axis=1)
        return scores


# --------------------------------------------------------------------------- #
# DIEN 的 AUGRU（Attention Update GRU）
# --------------------------------------------------------------------------- #
class AUGRU(layers.Layer):
    def __init__(self, units, **kwargs):
        super(AUGRU, self).__init__(**kwargs)
        self.units = units
        self.gru_cell = layers.GRUCell(units)

    def build(self, input_shape):
        # input_shape: [(batch, time, dim), (batch, time, 1)]
        self.gru_cell.build(input_shape[0])
        self.built = True

    def call(self, inputs, mask=None):
        seq_inputs, att_scores = inputs

        batch_size = tf.shape(seq_inputs)[0]
        seq_len = tf.shape(seq_inputs)[1]

        state = self.gru_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

        seq_inputs_t = tf.transpose(seq_inputs, [1, 0, 2])
        att_scores_t = tf.transpose(att_scores, [1, 0, 2])

        if mask is not None:
            mask_t = tf.transpose(mask, [1, 0])
        else:
            mask_t = tf.ones((seq_len, batch_size), dtype=tf.bool)

        kernel = self.gru_cell.kernel
        recurrent_kernel = self.gru_cell.recurrent_kernel
        input_bias, recurrent_bias = tf.unstack(self.gru_cell.bias)

        def step(prev_state, inputs_tuple):
            x, att, m = inputs_tuple

            matrix_x = tf.matmul(x, kernel)
            matrix_x = tf.nn.bias_add(matrix_x, input_bias)

            matrix_inner = tf.matmul(prev_state, recurrent_kernel)
            matrix_inner = tf.nn.bias_add(matrix_inner, recurrent_bias)

            x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)
            re_z, re_r, re_h = tf.split(matrix_inner, 3, axis=-1)

            z = tf.sigmoid(x_z + re_z)
            r = tf.sigmoid(x_r + re_r)
            hh = tf.tanh(x_h + r * re_h)

            # AUGRU：用 Attention 分数缩放更新门 z
            u = z * att

            new_h = (1 - u) * prev_state + u * hh

            m = tf.cast(m, dtype=tf.float32)
            m = tf.expand_dims(m, -1)
            final_h = m * new_h + (1 - m) * prev_state

            return final_h

        final_outputs = tf.scan(step, elems=(seq_inputs_t, att_scores_t, mask_t), initializer=state)

        return final_outputs[-1]


# --------------------------------------------------------------------------- #
# DMR 的位置编码层
# --------------------------------------------------------------------------- #
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.position_embeddings = self.add_weight(
            shape=(sequence_length, output_dim),
            initializer="uniform",
            name="position_embeddings",
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        position_embeddings = self.position_embeddings[:length, :]
        return inputs + position_embeddings


# --------------------------------------------------------------------------- #
# TWIN 的 Efficient Multi-Head Target Attention
# --------------------------------------------------------------------------- #
class EfficientMHTA(tf.keras.layers.Layer):
    def __init__(self, num_heads=2, key_dim=8, **kwargs):
        super(EfficientMHTA, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.output_dim = num_heads * key_dim

    def build(self, input_shape):
        self.W_q = tf.keras.layers.Dense(self.output_dim, use_bias=False, name="W_q")
        self.W_h = tf.keras.layers.Dense(self.output_dim, use_bias=False, name="W_h")
        self.W_v = tf.keras.layers.Dense(self.output_dim, use_bias=False, name="W_v")
        self.W_o = tf.keras.layers.Dense(self.output_dim, use_bias=False, name="W_o")
        super(EfficientMHTA, self).build(input_shape)

    def call(self, target, seq, mask=None):
        batch_size = tf.shape(target)[0]
        seq_len = tf.shape(seq)[1]

        target = tf.expand_dims(target, axis=1)  # [Batch, 1, Dim]

        q = self.W_q(target)
        k = self.W_h(seq)
        v = self.W_v(seq)

        q = tf.transpose(tf.reshape(q, [batch_size, 1, self.num_heads, self.key_dim]), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, [batch_size, seq_len, self.num_heads, self.key_dim]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, [batch_size, seq_len, self.num_heads, self.key_dim]), [0, 2, 1, 3])

        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)  # [Batch, 1, 1, Seq_len]
            padding_mask = (1.0 - mask) * -1e9
            scores += padding_mask

        weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(weights, v)  # [Batch, Num_Heads, 1, Key_dim]

        output = tf.reshape(tf.transpose(output, [0, 2, 1, 3]), [batch_size, 1, self.output_dim])
        output = tf.squeeze(output, axis=1)
        return self.W_o(output)


# --------------------------------------------------------------------------- #
# MIRRN 的点积 Target Attention（返回池化向量 + 分数）
# --------------------------------------------------------------------------- #
class DotTargetAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DotTargetAttention, self).__init__(**kwargs)

    def call(self, query, keys):
        query = tf.expand_dims(query, axis=1) if len(query.shape) == 2 else query
        attention_scores = tf.matmul(query, keys, transpose_b=True)  # [B, 1, S]
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        output = tf.matmul(attention_scores, keys)  # [B, 1, E]
        return tf.squeeze(output, axis=1), tf.squeeze(attention_scores, axis=1)


# --------------------------------------------------------------------------- #
# MIRRN 的 Multi-Head Fourier Transformer
# --------------------------------------------------------------------------- #
class MHFTLayer(keras.layers.Layer):
    def __init__(self, seq_len, emb_dim, **kwargs):
        super(MHFTLayer, self).__init__(**kwargs)
        self.S = seq_len
        self.E = emb_dim

    def build(self, input_shape):
        self.W_real = self.add_weight(name="w_real", shape=(self.S, self.E), initializer="glorot_uniform",
                                      trainable=True)
        self.W_imag = self.add_weight(name="w_imag", shape=(self.S, self.E), initializer="glorot_uniform",
                                      trainable=True)
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        super(MHFTLayer, self).build(input_shape)

    def call(self, x):
        x_t = tf.transpose(x, [0, 2, 1])  # [B, E, S]
        x_complex = tf.cast(x_t, tf.complex64)
        x_freq = tf.signal.fft(x_complex)
        x_freq = tf.transpose(x_freq, [0, 2, 1])  # [B, S, E]

        W_complex = tf.complex(self.W_real, self.W_imag)
        freq_out = x_freq * W_complex

        freq_out_t = tf.transpose(freq_out, [0, 2, 1])  # [B, E, S]
        x_out_complex = tf.signal.ifft(freq_out_t)
        x_out = tf.math.real(tf.transpose(x_out_complex, [0, 2, 1]))  # [B, S, E]

        return self.layer_norm(x + x_out)


# --------------------------------------------------------------------------- #
# FINAL 的因子化交互层与 FINAL Block
# --------------------------------------------------------------------------- #
class FactorizedInteractionLayer(keras.layers.Layer):
    def __init__(self, hidden_dim, num_steps=2, **kwargs):
        super(FactorizedInteractionLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.dense_layers = []

    def build(self, input_shape):
        for i in range(self.num_steps):
            self.dense_layers.append(
                keras.layers.Dense(self.hidden_dim, activation=None, name=f"final_step_{i}")
            )
        super(FactorizedInteractionLayer, self).build(input_shape)

    def call(self, x_prev):
        h = self.dense_layers[0](x_prev)
        outputs = [h]

        for i in range(1, self.num_steps):
            proj = self.dense_layers[i](x_prev)
            h = h * tf.nn.relu(proj)
            outputs.append(h)

        out = tf.add_n(outputs)
        return out


class FinalBlock(keras.layers.Layer):
    def __init__(self, hidden_dim, num_layers=2, num_steps=2, **kwargs):
        super(FinalBlock, self).__init__(**kwargs)
        self.interaction_layers = [
            FactorizedInteractionLayer(hidden_dim, num_steps=num_steps, name=f"fil_{i}")
            for i in range(num_layers)
        ]
        self.proj_out = keras.layers.Dense(1, activation=None, name="block_out")

    def call(self, x):
        out = x
        for layer in self.interaction_layers:
            out = layer(out)
        logit = self.proj_out(out)
        return logit


# --------------------------------------------------------------------------- #
# QNN-alpha 层（含 Multi-Head KRP 与 Mid-Act）
# --------------------------------------------------------------------------- #
class QNNAlphaLayer(keras.layers.Layer):
    def __init__(self, num_layers=2, num_heads=4, dropout_rate=0.1, **kwargs):
        super(QNNAlphaLayer, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.D = input_shape[-1]
        if self.D % self.num_heads != 0:
            raise ValueError(f"输入特征维度 {self.D} 必须能被多头数量 {self.num_heads} 整除。")
        self.head_dim = self.D // self.num_heads

        self.W_layers = []
        self.dropouts = []
        for i in range(self.num_layers):
            head_dense = []
            for h in range(self.num_heads):
                head_dense.append(keras.layers.Dense(self.head_dim, activation="relu", name=f"qnn_l{i}_h{h}"))
            self.W_layers.append(head_dense)
            self.dropouts.append(keras.layers.Dropout(self.dropout_rate))
        super(QNNAlphaLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.num_layers):
            x_heads = tf.split(x, self.num_heads, axis=-1)
            out_heads = []
            for h in range(self.num_heads):
                mid_act = self.W_layers[i][h](x)
                out_h = x_heads[h] * mid_act + x_heads[h]
                out_heads.append(out_h)
            x = tf.concat(out_heads, axis=-1)
            x = self.dropouts[i](x, training=training)
        return x

    def get_config(self):
        config = super(QNNAlphaLayer, self).get_config()
        config.update({
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config


# --------------------------------------------------------------------------- #
# DLF 的 Dynamic Low-Order-Aware Fusion 层
# --------------------------------------------------------------------------- #
class DLFLayer(keras.layers.Layer):
    def __init__(self, dim, num_heads=2, **kwargs):
        super(DLFLayer, self).__init__(**kwargs)
        self.dim = dim

        self.W_L_i = keras.layers.Dense(dim, use_bias=False)
        self.W_L_o = keras.layers.Dense(dim, activation="relu")

        self.W_C_i = keras.layers.Dense(dim, use_bias=False)
        self.W_C_o = keras.layers.Dense(dim, activation="relu")

        self.W_D = keras.layers.Dense(dim, activation="relu")

        self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.fuse_dense = keras.layers.Dense(dim, activation="relu")

        self.gate_dense = keras.layers.Dense(dim, use_bias=False)

    def call(self, inputs):
        E_1, E_l = inputs  # E_1: 第1层特征, E_l: 当前层特征

        Z_L = self.W_L_o(E_1 * self.W_L_i(E_l))
        Z_C = self.W_C_o(E_l * self.W_C_i(E_l))
        Z_D = self.W_D(E_l)

        stacked_Z = tf.stack([Z_L, Z_C, Z_D], axis=1)

        attn_out = self.mha(stacked_Z, stacked_Z)  # [Batch, 3, Dim]

        flat_attn = tf.reshape(attn_out, [-1, 3 * self.dim])
        Z_fused = self.fuse_dense(flat_attn)

        gate = self.gate_dense(E_l)
        gate = tf.maximum(1e-6, gate)  # epsilon

        E_next = Z_fused + gate * E_l

        return E_next


# --------------------------------------------------------------------------- #
# HierDiffuse 的 Target Attention 与融合层
# --------------------------------------------------------------------------- #
class MLPTargetAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MLPTargetAttention, self).__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(1, activation=None)

    def call(self, target_emb, seq_emb, mask=None):
        target_emb_expanded = tf.expand_dims(target_emb, axis=1)  # [B, 1, D]
        target_emb_tiled = tf.tile(target_emb_expanded, [1, tf.shape(seq_emb)[1], 1])  # [B, T, D]

        concat_info = tf.concat([target_emb_tiled, seq_emb, target_emb_tiled - seq_emb, target_emb_tiled * seq_emb],
                                axis=-1)

        attention_score = self.dense2(self.dense1(concat_info))  # [B, T, 1]

        if mask is not None:
            paddings = tf.ones_like(attention_score) * (-2 ** 32 + 1)
            mask_expanded = tf.expand_dims(mask, axis=-1)
            attention_score = tf.where(mask_expanded, attention_score, paddings)

        attention_weight = tf.nn.softmax(attention_score, axis=1)  # [B, T, 1]
        output = tf.reduce_sum(seq_emb * attention_weight, axis=1)  # [B, D]
        return output


class HierDiffuseFusion(keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(HierDiffuseFusion, self).__init__(**kwargs)
        self.dim = dim
        self.sgd_net = keras.layers.Dense(1, activation="sigmoid", name="sgd_weight")

        self.tcc_dense1 = keras.layers.Dense(dim * 2, activation="relu")
        self.tcc_dense2 = keras.layers.Dense(dim)

        self.gamma_dense = keras.layers.Dense(dim)
        self.beta_dense = keras.layers.Dense(dim)

    def call(self, long_rep, short_seq_emb, target_emb):
        target_expanded = tf.expand_dims(target_emb, axis=1)  # [B, 1, D]
        target_tiled = tf.tile(target_expanded, [1, tf.shape(short_seq_emb)[1], 1])
        sgd_input = tf.concat([short_seq_emb, target_tiled], axis=-1)
        omega_i = self.sgd_net(sgd_input)  # [B, T_s, 1]

        short_cond = tf.reduce_sum(short_seq_emb * omega_i, axis=1)  # [B, D]

        gamma = self.gamma_dense(short_cond)  # [B, D]
        beta = self.beta_dense(short_cond)  # [B, D]

        x_t = self.tcc_dense1(long_rep)
        x_t = self.tcc_dense2(x_t)

        fusion_rep = gamma * x_t + beta

        return fusion_rep + long_rep


# --------------------------------------------------------------------------- #
# ASENet 的 Mixture of Experts 层
# --------------------------------------------------------------------------- #
class MoELayer(keras.layers.Layer):
    def __init__(self, num_experts, expert_hidden_units, output_dim=1, **kwargs):
        super(MoELayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.output_dim = output_dim

        self.experts = []
        for i in range(self.num_experts):
            expert_layers = []
            for units in expert_hidden_units:
                expert_layers.append(keras.layers.Dense(units, activation="relu", name=f"expert_{i}_dense_{units}"))
            expert_layers.append(keras.layers.Dense(self.output_dim, name=f"expert_{i}_output"))
            self.experts.append(tf.keras.Sequential(expert_layers, name=f"expert_{i}"))

        self.gating_network = keras.layers.Dense(self.num_experts, activation="softmax", name="gating_network")

    def call(self, inputs):
        gate_weights = self.gating_network(inputs)

        mean_gate_weights = tf.reduce_mean(gate_weights, axis=0)
        load_balance_loss = self.num_experts * tf.reduce_sum(tf.square(mean_gate_weights)) - 1.0
        self.add_loss(load_balance_loss)

        expert_outputs = [expert(inputs) for expert in self.experts]
        stacked_expert_outputs = tf.stack(expert_outputs, axis=2)
        expanded_gate_weights = tf.expand_dims(gate_weights, axis=1)
        weighted_expert_outputs = stacked_expert_outputs * expanded_gate_weights
        final_output = tf.reduce_sum(weighted_expert_outputs, axis=2)
        return final_output
