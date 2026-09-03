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
# 特征们的字典：(最大编码数目，输入特征类型，嵌入维度，特征类型(0单值, 1序列)，是否选用，描述)
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
# 论文核心模块复现：DLF (Dynamic Low-Order-Aware Fusion)
# =========================================================================

class DLFLayer(keras.layers.Layer):
    def __init__(self, dim, num_heads=2, **kwargs):
        super(DLFLayer, self).__init__(**kwargs)
        self.dim = dim
        
        # 1. RLI: Low-Rank Interaction Block (结合初始特征 E_1)
        self.W_L_i = keras.layers.Dense(dim, use_bias=False)
        self.W_L_o = keras.layers.Dense(dim, activation='relu')
        
        # 2. RLI: High-Rank Interaction Block (当前层自交互)
        self.W_C_i = keras.layers.Dense(dim, use_bias=False)
        self.W_C_o = keras.layers.Dense(dim, activation='relu')
        
        # 3. RLI: Implicit Interaction Block (隐式全连接)
        self.W_D = keras.layers.Dense(dim, activation='relu')
        
        # 4. NAF: Network-Aware Attention Fusion Module (块间/块内注意力融合)
        # 用 MultiHeadAttention 实现块间的交叉注意力与自注意力
        self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim//num_heads)
        self.fuse_dense = keras.layers.Dense(dim, activation='relu')
        
        # 5. Gated Residual (门控残差机制)
        self.gate_dense = keras.layers.Dense(dim, use_bias=False)

    def call(self, inputs):
        E_1, E_l = inputs  # E_1: 第1层特征, E_l: 当前层特征
        
        # ========== RLI 模块 ==========
        # 低阶显式特征 (保持与E_1的交互，防止低阶信号丢失)
        Z_L = self.W_L_o(E_1 * self.W_L_i(E_l))
        
        # 高阶显式特征 (自身交互，提取高阶模式)
        Z_C = self.W_C_o(E_l * self.W_C_i(E_l))
        
        # 隐式特征 (非线性转换)
        Z_D = self.W_D(E_l)
        
        # ========== NAF 模块 ==========
        # 将三个特征块堆叠，形状变为 [Batch, 3, Dim]
        stacked_Z = tf.stack([Z_L, Z_C, Z_D], axis=1)
        
        # 利用 MHA 进行特征交互融合
        attn_out = self.mha(stacked_Z, stacked_Z) # [Batch, 3, Dim]
        
        # 展平后进行融合投影 -> [Batch, Dim]
        flat_attn = tf.reshape(attn_out, [-1, 3 * self.dim])
        Z_fused = self.fuse_dense(flat_attn)
        
        # ========== Gated Residual ==========
        # 动态控制保留多少上一层的残差信息，防止噪声冲突
        gate = self.gate_dense(E_l)
        gate = tf.maximum(1e-6, gate) # epsilon
        
        E_next = Z_fused + gate * E_l
        
        return E_next

class DLFModel(keras.Model):
    def __init__(self, features_dict, num_dlf_layers=2, **kwargs):
        super(DLFModel, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)
        
        # 将连续特征映射到与 embedding 相同的维度(8维)，以对齐维度
        self.num_proj = keras.layers.Dense(8, name='num_proj')
        
        # 计算打平后的总维度
        self.num_used_features = len([k for k, v in features_dict.items() if v[4]])
        self.flat_dim = self.num_used_features * 8 # 每个特征都是 8 维
        
        # 构建堆叠的 DLF 层
        self.dlf_layers = [DLFLayer(dim=self.flat_dim) for _ in range(num_dlf_layers)]
        
        # 最终预测层
        self.final_dense = keras.layers.Dense(1, name='dense_out')

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
                result[f] = keras.layers.Embedding(input_dim=features_dict[f][0], output_dim=features_dict[f][2], name=f + '_embedding')
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
            if self._features_dict[key][1] == tf.string: # 分类特征
                x_int = self._hash_dict.get(key)(formatted_inputs.get(key))
                x_emb = self._emb_dict.get(key)(x_int)
                if self._features_dict[key][3] == 1: # 序列特征
                    x_emb = tf.reduce_mean(x_emb, axis=-2)
                processed_inputs.append(x_emb)
            else: # 数值特征 (price)
                num_val = tf.expand_dims(formatted_inputs.get(key), -1) # [Batch] -> [Batch, 1]
                num_emb = self.num_proj(num_val)                        # [Batch, 1] -> [Batch, 8]
                processed_inputs.append(num_emb)
        
        # 将所有特征堆叠，然后展平
        # 结果维度: [Batch, num_features * 8] 也就是 E^(1)
        stacked_input = tf.stack(processed_inputs, axis=1)
        flat_input = tf.keras.layers.Flatten()(stacked_input)
        return flat_input

    @tf.function
    def call(self, data):
        data = self._format_inputs(data)
        
        # 获得初始特征表示 E^(1)
        E_1 = self._preprocess_features(data)
        
        E_l = E_1
        # 逐层经过 DLF 层
        for layer in self.dlf_layers:
            E_l = layer([E_1, E_l])
            
        # 输出预测概率
        logits = self.final_dense(E_l)
        y_pred = keras.activations.sigmoid(logits)
        return y_pred


input_spec = {}
for key in features_dict.keys():
    input_spec[key] = tf.TensorSpec(shape=[None, ], dtype=features_dict.get(key)[1], name=key)

# 实例化新的 DLF 模型 (默认堆叠 2 层 DLF)
model = DLFModel(features_dict, num_dlf_layers=4)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
auc = tf.keras.metrics.AUC(name='my_auc')

@tf.function
def serving_function(data):
    result = {'prob': model(data)}
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