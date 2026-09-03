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
# 特征们的字典，存储格式为(最大编码数目，输入特征类型，嵌入维度，特征类型（0为单值，1为序列），是否在本模型中选用，特征含义描述)
raw_features_dict['clk'] = (10, tf.float32, 8, 0, False, '是否点击， 训练的label')
raw_features_dict['btag_his'] = (10, tf.string, 8, 1, False, '行为类型序列，包括ipv/cart/fav/buy')
raw_features_dict['cate_his'] = (15000, tf.string, 8, 1, True, '历史点击广告商品的类别ID序列')
raw_features_dict['brand_his'] = (150000, tf.string, 8, 1, True, '历史点击广告商品的品牌ID序列')
raw_features_dict['userid'] = (1000000, tf.string, 8, 0, True, '用户ID')
raw_features_dict['cms_segid'] = (300, tf.string, 8, 0, True, '微群ID')
raw_features_dict['cms_group_id'] = (50, tf.string, 8, 0, True, '微群组ID')
raw_features_dict['final_gender_code'] = (10, tf.string, 8, 0, True, '性别')
raw_features_dict['age_level'] = (35, tf.string, 8, 0, True, '年龄层次')
raw_features_dict['pvalue_level'] = (20, tf.string, 8, 0, True, '消费档次， 低档/中档/高档')
raw_features_dict['shopping_level'] = (15, tf.string, 8, 0, True, '购物深度， 浅层/中度/深度')
raw_features_dict['occupation'] = (10, tf.string, 8, 0, True, '职业，是否大学生1：是，0：否')
raw_features_dict['new_user_class_level'] = (25, tf.string, 8, 0, True, '城市级别')
raw_features_dict['adgroup_id'] = (800000, tf.string, 8, 0, True, '广告ID')
raw_features_dict['cate_id'] = (15000, tf.string, 8, 0, True, '当前广告商品的类别ID')
raw_features_dict['campaign_id'] = (400000, tf.string, 8, 0, True, '广告计划ID')
raw_features_dict['customer'] = (300000, tf.string, 8, 0, True, '广告主ID')
raw_features_dict['brand'] = (150000, tf.string, 8, 0, True, '当前广告商品的品牌ID')
raw_features_dict['price'] = (300, tf.float32, 8, 0, True, '商品价格，已归一化 0～1')
raw_features_dict['pid'] = (10, tf.string, 8, 0, True, '资源位id')
raw_features_dict['btag'] = (10, tf.string, 8, 0, False, '行为类型，包括ipv/cart/fav/buy')

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

# =========================================================================
# [QNN-alpha 核心修改 1]: 定义 QNN-alpha 层 (包含 Multi-Head KRP 和 Mid-Act)
# =========================================================================
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
                # Mid-Act：在此处引入 ReLU 激活函数
                head_dense.append(keras.layers.Dense(self.head_dim, activation='relu', name=f'qnn_l{i}_h{h}'))
            self.W_layers.append(head_dense)
            self.dropouts.append(keras.layers.Dropout(self.dropout_rate))
        super(QNNAlphaLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.num_layers):
            # 1. 划分 Multi-Head
            x_heads = tf.split(x, self.num_heads, axis=-1)
            out_heads = []
            for h in range(self.num_heads):
                # 2. 线性投影并经过 Mid-Act (ReLU)
                mid_act = self.W_layers[i][h](x)
                # 3. KRP 交互 (Hadamard 形式) + 残差连接
                out_h = x_heads[h] * mid_act + x_heads[h]
                out_heads.append(out_h)
            # 4. 拼接多头输出并 Dropout
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


class CtrHashNN(keras.Model):
    def __init__(self, features_dict, **kwargs):
        super(CtrHashNN, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)
        
        # =========================================================================
        # [QNN-alpha 核心修改 2]: 替换原有的 MLP 为 QNN-alpha 结构
        # =========================================================================
        # 为了保证维度能被 num_heads 整除，先加一层映射层统一特征维度到 128
        self.project_layer = keras.layers.Dense(128, activation='relu', name='feature_projection')
        
        # 引入 QNN-alpha 网络 (论文推荐 2层到4层)
        self.qnn_alpha = QNNAlphaLayer(num_layers=1, num_heads=4, dropout_rate=0.1)
        
        # 最终输出层
        self.final_dense = keras.layers.Dense(1, activation='sigmoid', name='dense_out')

    def _init_hash_dict(self, features_dict):
        result = {}
        for f in features_dict.keys():
            if features_dict[f][1] == tf.string:
                result[f] = keras.layers.Hashing(num_bins=features_dict[f][0],name=f + '_hash')
        return result

    def _init_emb_dict(self, features_dict):
        result = {}
        for f in features_dict.keys():
            if features_dict[f][1] == tf.string:
                result[f] = keras.layers.Embedding(input_dim=features_dict[f][0], output_dim=features_dict[f][2],name=f + '_embedding')
        return result

    @tf.function
    def _format_inputs(self, inputs):
        features_recons = {}
        for key in self._features_dict.keys():
            if self._features_dict[key][3] == 1:
                features_recons[key] = tf.strings.split(tf.reshape(inputs.get(key), [-1]),sep='^')
            else:
                # 微调：增加一维以便后续能安全 concatenate
                features_recons[key] = tf.reshape(inputs.get(key), [-1, 1])
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
                else:
                    x_emb = tf.reshape(x_emb, [-1, self._features_dict[key][2]])
                processed_inputs.append(x_emb)
            else:
                processed_inputs.append(tf.cast(formatted_inputs.get(key), tf.float32))
        processed_input = tf.keras.layers.concatenate(processed_inputs, axis=-1)
        return processed_input

    @tf.function
    def call(self, data, training=False):
        data = self._format_inputs(data)
        x = self._preprocess_features(data)
        
        # =========================================================================
        # [QNN-alpha 核心修改 2.1]: 前向传播流转入 QNN-alpha 模块
        # =========================================================================
        x = self.project_layer(x)
        # 注意需要传递 training 标志位，控制 Dropout 生效 (为 SE Loss 做铺垫)
        x = self.qnn_alpha(x, training=training)
        y_pred = self.final_dense(x)
        return y_pred

    # =========================================================================
    # [QNN-alpha 核心修改 3]: 重写 train_step 实现 Self-Ensemble (SE) Loss
    # =========================================================================
    def train_step(self, data):
        # make_csv_dataset 返回的是 (features_dict, labels) 格式
        x, y = data
        
        with tf.GradientTape() as tape:
            # 1. 两次带有 Dropout 的前向传播
            y_pred1 = self(x, training=True)
            y_pred2 = self(x, training=True)
            
            # 为对数计算提供数值稳定性保护
            epsilon = 1e-7
            y_pred1 = tf.clip_by_value(y_pred1, epsilon, 1.0 - epsilon)
            y_pred2 = tf.clip_by_value(y_pred2, epsilon, 1.0 - epsilon)
            
            # 2. 计算预测的平均值 (Ensemble)
            y_pred_avg = (y_pred1 + y_pred2) / 2.0
            
            # 3. 基础损失：BCE Loss (使用平均预测值)
            bce_loss = self.compiled_loss(y, y_pred_avg, regularization_losses=self.losses)
            
            # 4. 论文提出的 SE Loss
            # 公式: - 1/N * sum[ y_tilde * log(y1*y2) + (1-y_tilde)*log((1-y1)*(1-y2)) ]
            y_tilde = tf.stop_gradient(y_pred_avg) # 冻结软标签梯度
            se_loss = -tf.reduce_mean(
                y_tilde * tf.math.log(y_pred1 * y_pred2 + epsilon) + 
                (1 - y_tilde) * tf.math.log((1 - y_pred1) * (1 - y_pred2) + epsilon)
            )
            
            # 总损失
            total_loss = bce_loss + se_loss

        # 梯度更新
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # 更新 Metrics
        self.compiled_metrics.update_state(y, y_pred_avg)
        return {m.name: m.result() for m in self.metrics}

    # 重写 test_step 确保预测时单次前向不出问题
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


input_spec = {}
for key in features_dict.keys():
    input_spec[key] = tf.TensorSpec(shape=[None, ], dtype=features_dict.get(key)[1], name=key)

model = CtrHashNN(features_dict)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
auc = tf.keras.metrics.AUC(name='my_auc')

@tf.function
def serving_function(data):
    # 预测服务调用时，training默认为False，单次前向无延迟负担
    result = {'prob': model(data, training=False)}
    return result

output_func = serving_function.get_concrete_function(input_spec)
model.compile(optimizer=optimizer, loss=loss, metrics=[auc])
history = model.fit(train_dataset, epochs=EPOCHS, verbose=1, validation_data=valid_dataset)

current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
print('current time: ', current_time)
model.save(os.path.join(serving_model_path, current_time), overwrite=True, include_optimizer=False,
           signatures={tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: output_func})

new_model = tf.keras.models.load_model(os.path.join(serving_model_path, current_time))
pre_cvr_list = new_model.predict(test_dataset)

def bylineread(filename):
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            yield line

read = bylineread(test_files)
labels = [int(line.strip().split(',')[label_idx]) for line in read]

if len(labels) > 0 and len(pre_cvr_list) > 0:
    test_losloss = log_loss(labels, pre_cvr_list)
    test_auc = roc_auc_score(labels, pre_cvr_list)
    test_pcoc = sum(pre_cvr_list)[0] / sum(labels) - 1 if sum(labels) > 0 else 0
    print("test_losloss:{:.6f}, test_auc:{:.6f}, test_pcoc:{:.6f}".format(test_losloss, test_auc, test_pcoc))
else:
    print("Test data is empty, skipping evaluation.")

final_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('final time: ', final_time)