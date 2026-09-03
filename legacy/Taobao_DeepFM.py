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

# 设置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np.random.seed(42)
tf.random.set_seed(42)

# ================= 数据定义部分 (保持不变) =================
raw_features_dict = OrderedDict()
raw_features_dict['clk'] = (10, tf.float32, 8, 0, False, '是否点击， 训练的label')
raw_features_dict['btag_his'] = (10, tf.string, 8, 1, False, '行为类型序列，包括ipv/cart/fav/buy')
raw_features_dict['cate_his'] = (15000, tf.string, 8, 1, False, '历史点击广告商品的类别ID序列')
raw_features_dict['brand_his'] = (200000, tf.string, 8, 1, False, '历史点击广告商品的品牌ID序列')
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

# 请修改为您实际的路径
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


# ================= 模型定义部分 (修改为 DeepFM) =================

class DeepFM(keras.Model):
    def __init__(self, features_dict, **kwargs):
        super(DeepFM, self).__init__(**kwargs)
        self._features_dict = features_dict
        
        # 1. 初始化 Hashing 层
        self._hash_dict = self._init_hash_dict(features_dict)
        
        # 2. 初始化 Embedding 层 (用于 FM二阶交叉 和 Deep 部分) - 维度为 k (这里是8)
        self._emb_dict = self._init_emb_dict(features_dict)
        
        # 3. 初始化 Linear Embedding 层 (用于 FM一阶线性部分) - 维度为 1
        # 对于 ID 特征，使用 Output=1 的 Embedding 模拟权重 w
        self._linear_emb_dict = self._init_linear_emb_dict(features_dict)
        
        # 对于数值特征 (如 price)，使用 Dense(1) 模拟权重 w * x
        self._dense_linear_layer = keras.layers.Dense(1, activation=None, use_bias=False, name='dense_linear')

        # 4. Deep 部分 (DNN)
        self._deep_net = tf.keras.Sequential([
            keras.layers.Dense(128, name='deep_dense_1', activation='relu'),
            keras.layers.Dense(64, name='deep_dense_2', activation='relu'),
            keras.layers.Dense(1, name='deep_out')
        ])

    def _init_hash_dict(self, features_dict):
        result = {}
        for f, spec in features_dict.items():
            if spec[1] == tf.string or f == 'price': # 字符串类型需要Hash
                result[f] = keras.layers.Hashing(num_bins=spec[0], name=f + '_hash')
        return result

    def _init_emb_dict(self, features_dict):
        # FM二阶和Deep共享的Embedding矩阵 (Dim=8)
        result = {}
        for f, spec in features_dict.items():
            if spec[1] == tf.string or f == 'price':
                result[f] = keras.layers.Embedding(input_dim=spec[0], output_dim=spec[2], name=f + '_embedding')
        return result

    def _init_linear_emb_dict(self, features_dict):
        # FM一阶Embedding (Dim=1)，相当于 Bias 或 Weight
        result = {}
        for f, spec in features_dict.items():
            if spec[1] == tf.string or f == 'price':
                result[f] = keras.layers.Embedding(input_dim=spec[0], output_dim=1, name=f + '_linear_embedding')
        return result

    @tf.function
    def _format_inputs(self, inputs):
        features_recons = {}
        for key in self._features_dict.keys():
            # 序列特征分割逻辑 (如 "A^B^C" -> ["A", "B", "C"])
            if self._features_dict[key][3] == 1:
                features_recons[key] = tf.strings.split(tf.reshape(inputs.get(key), [-1]), sep='^')
            else:
                features_recons[key] = tf.reshape(inputs.get(key), [-1])
        return features_recons

    @tf.function
    def call(self, data):
        data = self._format_inputs(data)
        
        # 容器
        sparse_embeds = []    # 存储 (Batch, k=8) 的向量，用于FM二阶和Deep
        linear_terms = []     # 存储 (Batch, 1) 的标量，用于FM一阶
        dense_inputs = []     # 存储数值特征，直接用于Deep

        for key, spec in self._features_dict.items():
            input_val = data.get(key)
            
            # --- 处理离散/序列特征 (String) ---
            if spec[1] == tf.string:
                hashed_val = self._hash_dict[key](input_val)
                
                # A. 计算 Linear Part (一阶)
                linear_emb = self._linear_emb_dict[key](hashed_val) # (Batch, 1) 或 (Batch, Seq, 1)
                if spec[3] == 1: # 如果是序列，取平均作为该域的贡献
                    linear_emb = tf.reduce_mean(linear_emb, axis=-2)
                linear_terms.append(linear_emb)
                
                # B. 计算 Interaction/Deep Part (Embedding)
                emb = self._emb_dict[key](hashed_val) # (Batch, 8) 或 (Batch, Seq, 8)
                if spec[3] == 1: # 如果是序列，取平均作为该域的Embedding向量
                    emb = tf.reduce_mean(emb, axis=-2)
                sparse_embeds.append(emb)
            elif key == 'price' and spec[1] == tf.float32:
                # price 特征的离散化和嵌入逻辑
                price_scaled = data.get(key) * 100.0
                price_binned = tf.cast(price_scaled, dtype=tf.int32)
                price_str = tf.strings.as_string(price_binned)
                hashed_val = self._hash_dict.get(key)(price_str)
                linear_emb = self._linear_emb_dict[key](hashed_val)
                linear_terms.append(linear_emb)
                emb = self._emb_dict[key](hashed_val)
                sparse_embeds.append(emb)
            
            # --- 处理数值特征 (Float) ---
            else:
                # 数值特征 reshape 成 (Batch, 1)
                num_val = tf.reshape(input_val, [-1, 1])
                
                # A. Linear Part: w * x
                linear_terms.append(self._dense_linear_layer(num_val))
                
                # B. Deep Part: 直接拼接到 Dense
                dense_inputs.append(num_val)
                # 注意：数值特征通常不参与 FM 的二阶 ID 交叉，除非进行分桶Embedding。这里遵循常规做法仅放入Deep和Linear。

        # ------------------------------------------------------
        # 1. FM First Order (Linear)
        # Sum(w*x + b) -> 对应所有 linear_terms 的求和
        linear_logit = tf.add_n(linear_terms) # (Batch, 1)

        # ------------------------------------------------------
        # 2. FM Second Order (Interaction)
        # Formula: 0.5 * sum( (sum(v_i)^2) - sum(v_i^2) )
        # 将所有 Sparse Feature 的 Embedding 堆叠: (Batch, Num_Fields, Emb_Dim)
        if len(sparse_embeds) > 0:
            fm_input_emb = tf.stack(sparse_embeds, axis=1) # (Batch, N, 8)
            
            sum_square = tf.square(tf.reduce_sum(fm_input_emb, axis=1)) # (sum(v))^2 -> (Batch, 8)
            square_sum = tf.reduce_sum(tf.square(fm_input_emb), axis=1) # sum(v^2)    -> (Batch, 8)
            
            # 二阶交叉结果 (Batch, 1)
            fm_second_order = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1, keepdims=True)
        else:
            fm_second_order = 0.0

        # ------------------------------------------------------
        # 3. Deep Component
        # 将 Embedding 和 数值特征拼接
        deep_emb_flatten = tf.keras.layers.Flatten()(tf.stack(sparse_embeds, axis=1)) if len(sparse_embeds) > 0 else None
        
        if deep_emb_flatten is not None and len(dense_inputs) > 0:
            deep_input = tf.concat([deep_emb_flatten] + dense_inputs, axis=-1)
        elif deep_emb_flatten is not None:
            deep_input = deep_emb_flatten
        else:
            deep_input = tf.concat(dense_inputs, axis=-1)
            
        deep_logit = self._deep_net(deep_input)

        # ------------------------------------------------------
        # Final Output
        # DeepFM = Sigmoid(Linear + FM_Interaction + Deep)
        total_logit = linear_logit + fm_second_order + deep_logit
        y_pred = tf.nn.sigmoid(total_logit)
        
        return y_pred


# ================= 训练与评估部分 (保持不变) =================

input_spec = {}
for key in features_dict.keys():
    input_spec[key] = tf.TensorSpec(shape=[None, ], dtype=features_dict.get(key)[1], name=key)

# 实例化新的 DeepFM 模型
model = DeepFM(features_dict)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
auc = tf.keras.metrics.AUC(name='my_auc')

@tf.function
def serving_function(data):
    result = {'prob': model(data)}
    return result

output_func = serving_function.get_concrete_function(input_spec)
model.compile(optimizer=optimizer, loss=loss, metrics=[auc])

# 训练
history = model.fit(train_dataset, epochs=EPOCHS, verbose=1, validation_data=valid_dataset)

predictions = model.predict(test_dataset)


def bylineread(filename):
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            yield line


read = bylineread(test_files)
labels = [int(line.strip().split(',')[label_idx]) for line in read]

if len(labels) > 0 and len(predictions) > 0:
    test_losloss = log_loss(labels, predictions)
    test_auc = roc_auc_score(labels, predictions)
    test_pcoc = sum(predictions)[0] / sum(labels) - 1 if sum(labels) > 0 else 0
    print("test_losloss:{:.6f}, test_auc:{:.6f}, test_pcoc:{:.6f}".format(test_losloss, test_auc, test_pcoc))
else:
    print("Test data is empty, skipping evaluation.")


@tf.function
def serving_function(data):
    result = {'prob': model(data)}
    return result

input_spec = {}
for key in features_dict.keys():
    input_spec[key] = tf.TensorSpec(shape=[None, ], dtype=features_dict.get(key)[1], name=key)

output_func = serving_function.get_concrete_function(input_spec)

current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
print('current time: ', current_time)
model.save(os.path.join(serving_model_path, current_time), overwrite=True, include_optimizer=False,
           signatures={tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: output_func})

final_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('final time: ', final_time)