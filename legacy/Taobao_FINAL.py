from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import tensorflow as tf
import numpy as np
import os
from collections import OrderedDict
import tensorflow.keras as keras
from datetime import date, timedelta
import time
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np.random.seed(42)
tf.random.set_seed(42)

raw_features_dict = OrderedDict()
raw_features_dict['clk'] = (10, tf.float32, 8, 0, False, '是否点击， 训练的label')
raw_features_dict['btag_his'] = (10, tf.string, 8, 1, False, '行为类型序列，包括ipv/cart/fav/buy')
raw_features_dict['cate_his'] = (15000, tf.string, 8, 1, True, '历史点击广告商品的类别ID序列')
raw_features_dict['brand_his'] = (200000, tf.string, 8, 1, True, '历史点击广告商品的品牌ID序列')
raw_features_dict['userid'] = (1500000, tf.string, 8, 0, True, '用户ID')
raw_features_dict['cms_segid'] = (300, tf.string, 8, 0, True, '微群ID')
raw_features_dict['cms_group_id'] = (50, tf.string, 8, 0, True, '微群组ID')
raw_features_dict['final_gender_code'] = (10, tf.string, 8, 0, True, '性别')
raw_features_dict['age_level'] = (35, tf.string, 8, 0, True, '年龄层次')
raw_features_dict['pvalue_level'] = (20, tf.string, 8, 0, True, '消费档次， 低档/中档/高档')
raw_features_dict['shopping_level'] = (15, tf.string, 8, 0, True, '购物深度， 浅层/中度/深度')
raw_features_dict['occupation'] = (10, tf.string, 8, 0, True, '职业，是否大学生1：是，0：否')
raw_features_dict['new_user_class_level'] = (25, tf.string, 8, 0, True, '城市级别')
raw_features_dict['adgroup_id'] = (1200000, tf.string, 8, 0, True, '广告ID')
raw_features_dict['cate_id'] = (15000, tf.string, 8, 0, True, '当前广告商品的类别ID')
raw_features_dict['campaign_id'] = (800000, tf.string, 8, 0, True, '广告计划ID')
raw_features_dict['customer'] = (500000, tf.string, 8, 0, True, '广告主ID')
raw_features_dict['brand'] = (200000, tf.string, 8, 0, True, '当前广告商品的品牌ID')
raw_features_dict['price'] = (300, tf.float32, 8, 0, True, '商品价格，已归一化 0～1')
raw_features_dict['pid'] = (10, tf.string, 8, 0, True, '资源位id')
raw_features_dict['btag'] = (10, tf.string, 8, 0, False, '行为类型，包括ipv/cart/fav/buy')

# --- 3. 数据加载配置 ---
label_col = 'clk'

if label_col == 'clk':
    label_idx = 0
else:
    label_idx = -1

_CSV_COLUMNS = []
_SELECT_COLUMNS = []
_COLUMN_DEFAULTS = []
features_dict = OrderedDict()
for key, value in raw_features_dict.items():
    _CSV_COLUMNS.append(key)
    if value[4] == True:
        features_dict[key] = value
    if value[4] == True or key == label_col:
        _SELECT_COLUMNS.append(key)
        if value[1] is tf.int32:
            _COLUMN_DEFAULTS.append(tf.constant(0, dtype=value[1]))
        elif value[1] is tf.float32:
            _COLUMN_DEFAULTS.append(tf.constant(0.0, dtype=value[1]))
        else:
            _COLUMN_DEFAULTS.append(tf.constant("", dtype=value[1]))

_SHUFFLE_SIZE = 100000
print("_CSV_COLUMNS：{}, _SELECT_COLUMNS:{}".format(_CSV_COLUMNS, _SELECT_COLUMNS), len(_CSV_COLUMNS),
      len(_SELECT_COLUMNS))


def get_dataset(data_path, shuffle=True, num_epochs=1, batch_size=512, label_col=label_col):
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=data_path,
        batch_size=batch_size,
        column_names=_CSV_COLUMNS,
        column_defaults=_COLUMN_DEFAULTS,
        label_name=label_col,
        select_columns=_SELECT_COLUMNS,
        field_delim=',',
        header=True,
        num_epochs=num_epochs,
        shuffle=shuffle,
        shuffle_buffer_size=_SHUFFLE_SIZE,
        prefetch_buffer_size=tf.data.AUTOTUNE,
        num_parallel_reads=tf.data.AUTOTUNE,
        shuffle_seed=42
    )
    return dataset


# 请确保路径正确
train_files = '/data2/jupyter/nobackup/dataset/searchv6/train_tb.csv'
valid_files = '/data2/jupyter/nobackup/dataset/searchv6/valid_tb.csv'
test_files = '/data2/jupyter/nobackup/dataset/searchv6/test.csv'
serving_model_path = '/data2/jupyter/nobackup/model/search_ctr_gan_v1/'

BATCH_SIZE = 2048
EPOCHS = 1
LEARNING_RATE = 0.001

train_dataset = get_dataset(train_files, shuffle=True, num_epochs=1, batch_size=BATCH_SIZE)
valid_dataset = get_dataset(valid_files, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)
test_dataset = get_dataset(test_files, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)


# ==================== FINAL 核心架构定义 ====================

class FactorizedInteractionLayer(keras.layers.Layer):
    """
    论文中的单层 FINAL Layer:
    h_{l,1} = W_{l,1} * x_{l-1} + b_{l,1}
    h_{l,i} = h_{l,i-1} * \sigma(W_{l,i} * x_{l-1} + b_{l,i})
    x_l = \sum h_{l,i}
    """

    def __init__(self, hidden_dim, num_steps=2, **kwargs):
        super(FactorizedInteractionLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.dense_layers = []

    def build(self, input_shape):
        for i in range(self.num_steps):
            self.dense_layers.append(
                keras.layers.Dense(self.hidden_dim, activation=None, name=f'final_step_{i}')
            )
        super(FactorizedInteractionLayer, self).build(input_shape)

    def call(self, x_prev):
        # step 1: linear transformation
        h = self.dense_layers[0](x_prev)
        outputs = [h]

        # step 2 to N: Multiplicative interactions
        for i in range(1, self.num_steps):
            proj = self.dense_layers[i](x_prev)
            # 使用 ReLU 作为激活函数 \sigma，然后进行哈达玛积 (Hadamard Product)
            h = h * tf.nn.relu(proj)
            outputs.append(h)

        # 聚合所有阶的特征交叉结果
        out = tf.add_n(outputs)
        return out


class FinalBlock(keras.layers.Layer):
    """
    论文中的 FINAL Block，堆叠多个 FactorizedInteractionLayer 以实现指数级多项式交叉
    """

    def __init__(self, hidden_dim, num_layers=2, num_steps=2, **kwargs):
        super(FinalBlock, self).__init__(**kwargs)
        self.interaction_layers = [
            FactorizedInteractionLayer(hidden_dim, num_steps=num_steps, name=f'fil_{i}')
            for i in range(num_layers)
        ]
        self.proj_out = keras.layers.Dense(1, activation=None, name='block_out')

    def call(self, x):
        out = x
        for layer in self.interaction_layers:
            out = layer(out)
        logit = self.proj_out(out)
        return logit


class CtrFINALModel(keras.Model):
    def __init__(self, features_dict, hidden_dim=128, num_blocks=2, **kwargs):
        super(CtrFINALModel, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)

        self.num_blocks = num_blocks
        # 论文精华：构建并行的多个(默认2个) FINAL Blocks
        # 遵循论文实现细节 K=2, N=2
        self.final_blocks = [
            FinalBlock(hidden_dim=hidden_dim, num_layers=2, num_steps=2, name=f'final_block_{i}')
            for i in range(self.num_blocks)
        ]

    def _init_hash_dict(self, features_dict):
        result = {}
        for f in features_dict.keys():
            if features_dict[f][1] == tf.string:
                result[f] = keras.layers.Hashing(num_bins=features_dict[f][0], name=f + '_hash')
        return result

    def _init_emb_dict(self, features_dict):
        result = {}
        for f in features_dict.keys():
            if features_dict[f][1] == tf.string:
                result[f] = keras.layers.Embedding(input_dim=features_dict[f][0], output_dim=features_dict[f][2],
                                                   name=f + '_embedding')
        return result

    @tf.function
    def _format_inputs(self, inputs):
        features_recons = {}
        for key in self._features_dict.keys():
            if self._features_dict[key][3] == 1:
                features_recons[key] = tf.strings.split(tf.reshape(inputs.get(key), [-1]), sep='^')
            else:
                features_recons[key] = tf.reshape(inputs.get(key), [-1])
        return features_recons

    @tf.function
    def _preprocess_features(self, formatted_inputs):
        processed_inputs = []
        for key in self._features_dict.keys():
            if self._features_dict[key][1] == tf.string:
                x_int = self._hash_dict.get(key)(formatted_inputs.get(key))
                x_emb = self._emb_dict.get(key)(x_int)
                if self._features_dict[key][3] == 1:
                    x_emb = tf.reduce_mean(x_emb, axis=-2)
                processed_inputs.append(x_emb)
            else:
                # 修复原版连续特征拼接的潜在bug：从(B,) 转换为 (B,1) 并保持数据类型一致
                val = formatted_inputs.get(key)
                val = tf.cast(val, tf.float32)
                val = tf.reshape(val, [-1, 1])
                processed_inputs.append(val)

        processed_input = tf.keras.layers.concatenate(processed_inputs, axis=-1)
        return processed_input

    def call(self, data, training=False):
        data = self._format_inputs(data)
        x = self._preprocess_features(data)

        # 并行通过多个 FINAL Blocks 得到多个 logits
        block_logits = [block(x) for block in self.final_blocks]

        # 聚合（Average）为统一的 logit
        if self.num_blocks > 1:
            concat_logits = tf.concat(block_logits, axis=-1)
            final_logit = tf.reduce_mean(concat_logits, axis=-1, keepdims=True)
        else:
            final_logit = block_logits[0]

        y_pred = keras.activations.sigmoid(final_logit)

        # 如果是训练阶段且具有多个Block，返回所有Block独立预测的分数用于知识蒸馏
        if training and self.num_blocks > 1:
            block_preds = [keras.activations.sigmoid(l) for l in block_logits]
            return y_pred, block_preds

        return y_pred

    # 重写 train_step 以支持论文中的 "Cross-block Knowledge Transfer" (自知识蒸馏)
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            if self.num_blocks > 1:
                y_pred, block_preds = self(x, training=True)
                # 1. 主任务损失 CTR Loss (公式2)
                task_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

                # 2. 跨Block蒸馏损失 Distillation Loss (公式3)
                # 将汇总后的 y_pred 作为 teacher (必须 stop_gradient 切断梯度回传)，独立预测作为 student
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


# ==================== 实例化训练与线上导出 ====================

input_spec = {}
for key in features_dict.keys():
    input_spec[key] = tf.TensorSpec(shape=[None, ], dtype=features_dict.get(key)[1], name=key)

# 使用双流 FINAL 架构
model = CtrFINALModel(features_dict, hidden_dim=128, num_blocks=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
auc = tf.keras.metrics.AUC(name='my_auc')


@tf.function
def serving_function(data):
    # Serving 阶段默认 training=False，仅返回单体聚合预测概率
    result = {'prob': model(data, training=False)}
    return result


output_func = serving_function.get_concrete_function(input_spec)
model.compile(optimizer=optimizer, loss=loss, metrics=[auc])

print("开始训练 FINAL 模型...")
history = model.fit(train_dataset, epochs=EPOCHS, verbose=1, validation_data=valid_dataset)

current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
print('current time: ', current_time)
model.save(os.path.join(serving_model_path, current_time), overwrite=True, include_optimizer=False,
           signatures={tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: output_func})

print("开始评估...")
new_model = tf.keras.models.load_model(os.path.join(serving_model_path, current_time))
pre_cvr_list = new_model.predict(test_dataset)


def bylineread(filename):
    with open(filename, 'r') as f:
        next(f)  # 跳过表头
        for line in f:
            yield line


read = bylineread(test_files)
labels = [int(line.strip().split(',')[0]) for line in read]

if len(labels) > 0 and len(pre_cvr_list) > 0:
    test_losloss = log_loss(labels, pre_cvr_list)
    test_auc = roc_auc_score(labels, pre_cvr_list)
    test_pcoc = sum(pre_cvr_list)[0] / sum(labels) - 1 if sum(labels) > 0 else 0
    print("test_losloss:{:.6f}, test_auc:{:.6f}, test_pcoc:{:.6f}".format(test_losloss, test_auc, test_pcoc))
else:
    print("Test data is empty, skipping evaluation.")

final_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('final time: ', final_time)
