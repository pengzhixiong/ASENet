from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import tensorflow as tf
import numpy as np
import os
from collections import OrderedDict
import tensorflow.keras as keras
from tensorflow.keras import layers
from datetime import date, timedelta
import time
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# ================= 配置参数 =================
MAX_SEQ_LEN = 50  # 固定序列长度
BATCH_SIZE = 2048
EPOCHS = 1
LEARNING_RATE = 0.001

# ================= 特征定义 =================
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

# DIEN 序列对齐配置
SEQUENCE_PAIRS = {
    'cate_his': 'cate_id',
    'brand_his': 'brand'
}

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

# ================= 数据读取 =================
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

train_dataset = get_dataset(train_files, shuffle=True, num_epochs=1, batch_size=BATCH_SIZE)
valid_dataset = get_dataset(valid_files, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)
test_dataset = get_dataset(test_files, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)


# ================= DIEN 核心组件 (修复版) =================

class AUGRU(layers.Layer):
    """
    Attention Update GRU (AUGRU)
    修复了 bias 形变导致的问题
    """
    def __init__(self, units, **kwargs):
        super(AUGRU, self).__init__(**kwargs)
        self.units = units
        self.gru_cell = layers.GRUCell(units)

    def build(self, input_shape):
        # input_shape: [(batch, time, dim), (batch, time, 1)]
        # 手动 build gru_cell 以确保权重被创建
        self.gru_cell.build(input_shape[0])
        self.built = True

    def call(self, inputs, mask=None):
        """
        inputs: [gru_input_seq, attention_scores]
        """
        seq_inputs, att_scores = inputs
        
        batch_size = tf.shape(seq_inputs)[0]
        seq_len = tf.shape(seq_inputs)[1]
        
        state = self.gru_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        
        # 维度转置为 (Time, Batch, Dim) 以适配 tf.scan
        seq_inputs_t = tf.transpose(seq_inputs, [1, 0, 2])
        att_scores_t = tf.transpose(att_scores, [1, 0, 2])
        
        if mask is not None:
            mask_t = tf.transpose(mask, [1, 0])
        else:
            mask_t = tf.ones((seq_len, batch_size), dtype=tf.bool)

        # 【核心修复】：在 Loop 之外提取权重并进行 Bias 拆分
        # GRUCell.bias 形状通常是 [2, 3*units]
        kernel = self.gru_cell.kernel
        recurrent_kernel = self.gru_cell.recurrent_kernel
        # 拆分 bias，bias[0] 是 input bias, bias[1] 是 recurrent bias
        input_bias, recurrent_bias = tf.unstack(self.gru_cell.bias)

        def step(prev_state, inputs_tuple):
            x, att, m = inputs_tuple 
            
            # 手动实现 GRU 计算步骤
            # 1. 线性变换
            matrix_x = tf.matmul(x, kernel)
            matrix_x = tf.nn.bias_add(matrix_x, input_bias) # 这里 input_bias 已经是 Rank 1
            
            matrix_inner = tf.matmul(prev_state, recurrent_kernel)
            matrix_inner = tf.nn.bias_add(matrix_inner, recurrent_bias) # 这里 recurrent_bias 已经是 Rank 1
            
            # 2. 分割门控 (z: update, r: reset, h: candidate)
            x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)
            re_z, re_r, re_h = tf.split(matrix_inner, 3, axis=-1)
            
            # 3. 激活函数
            z = tf.sigmoid(x_z + re_z)
            r = tf.sigmoid(x_r + re_r)
            hh = tf.tanh(x_h + r * re_h)
            
            # 4. AUGRU 逻辑：利用 Attention 分数缩放更新门 z
            # att shape: (B, 1) -> 广播到 (B, units)
            u = z * att 
            
            # 5. 状态更新
            new_h = (1 - u) * prev_state + u * hh
            
            # 6. Mask 处理
            m = tf.cast(m, dtype=tf.float32)
            m = tf.expand_dims(m, -1)
            final_h = m * new_h + (1 - m) * prev_state
            
            return final_h

        # 使用 scan
        final_outputs = tf.scan(step, elems=(seq_inputs_t, att_scores_t, mask_t), initializer=state)
        
        return final_outputs[-1]


class AttentionLayer(layers.Layer):
    def __init__(self, hidden_units=(32, 16), activation='sigmoid', **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.dense_layers = []

    def build(self, input_shape):
        for unit in self.hidden_units:
            self.dense_layers.append(layers.Dense(unit, activation=self.activation))
        self.out_dense = layers.Dense(1, activation=None) 
        self.built = True

    def call(self, inputs, mask=None):
        keys, query = inputs # keys: History, query: Target
        
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

class DIEN(keras.Model):
    def __init__(self, features_dict, seq_pairs, **kwargs):
        super(DIEN, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._seq_pairs = seq_pairs
        
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)
        
        seq_emb_dim = 0
        for seq_name in self._seq_pairs.keys():
            seq_emb_dim += features_dict[seq_name][2]
            
        self.gru_extractor = layers.GRU(seq_emb_dim, return_sequences=True, name="interest_extractor")
        self.attention = AttentionLayer(name="attention_layer")
        self.augru = AUGRU(seq_emb_dim, name="interest_evolving")
        
        self.bn = layers.BatchNormalization()
        self.dense1 = layers.Dense(128, activation='relu', name='dense_1')
        self.dense2 = layers.Dense(64, activation='relu', name='dense_2')
        self.final_dense = layers.Dense(1, activation=None, name='dense_out')

    def _init_hash_dict(self, features_dict):
        result = {}
        for f in features_dict.keys():
            if features_dict[f][1] == tf.string or f == 'price':
                result[f] = keras.layers.Hashing(num_bins=features_dict[f][0], name=f + '_hash')
        return result

    def _init_emb_dict(self, features_dict):
        result = {}
        for f in features_dict.keys():
            if features_dict[f][1] == tf.string or f == 'price':
                result[f] = keras.layers.Embedding(input_dim=features_dict[f][0], 
                                                   output_dim=features_dict[f][2],
                                                   mask_zero=True, 
                                                   name=f + '_embedding')
        return result

    @tf.function
    def _format_inputs(self, inputs):
        features_recons = {}
        for key in self._features_dict.keys():
            val = inputs.get(key)
            if self._features_dict[key][3] == 1:
                split_str = tf.strings.split(val, sep='^')
                # 截断或填充到 MAX_SEQ_LEN (50)
                dense_str = split_str.to_tensor(default_value='0', shape=[None, MAX_SEQ_LEN])
                features_recons[key] = dense_str
            else:
                features_recons[key] = tf.reshape(val, [-1, 1])
        return features_recons

    @tf.function
    def call(self, data):
        data = self._format_inputs(data)
        
        other_features_emb = []
        seq_emb_list = []
        target_emb_list = []
        mask = None
        
        for key in self._features_dict.keys():
            feature_conf = self._features_dict[key]
            is_seq = feature_conf[3] == 1
            dtype = feature_conf[1]
            
            if dtype == tf.string:
                hashed = self._hash_dict[key](data[key])
                emb = self._emb_dict[key](hashed)
            elif dtype == tf.float32 and key == 'price':
                # price 特征的离散化和嵌入逻辑
                price_scaled = data.get(key) * 100.0
                price_binned = tf.cast(price_scaled, dtype=tf.int32)
                price_str = tf.strings.as_string(price_binned)
                hashed_feature = self._hash_dict.get(key)(price_str)
                emb = self._emb_dict.get(key)(hashed_feature)
            else:
                emb = data[key]
                if len(emb.shape) == 2 and not is_seq:
                    emb = tf.expand_dims(emb, -1)

            if is_seq:
                seq_emb_list.append(emb)
                if mask is None:
                     mask = tf.not_equal(hashed, 0)
            elif key in self._seq_pairs.values():
                target_emb_list.append(emb)
            else:
                if len(emb.shape) == 3:
                    emb = tf.reduce_sum(emb, axis=1)
                other_features_emb.append(emb)

        # 拼接 Embeddings
        # 注意：这里假设 raw_features_dict 中的定义顺序能够保证 seq_emb_list 和 target_emb_list 
        # 的顺序是对应上的。
        history_seq_emb = tf.concat(seq_emb_list, axis=-1)
        target_item_emb = tf.concat(target_emb_list, axis=-1)

        # 1. 兴趣抽取
        gru_out = self.gru_extractor(history_seq_emb, mask=mask)
        
        # 2. Attention
        att_scores = self.attention([gru_out, target_item_emb], mask=mask)
        
        # 3. 兴趣进化 (AUGRU)
        final_interest = self.augru([gru_out, att_scores], mask=mask)
        
        # 4. MLP
        target_item_emb_sq = tf.squeeze(target_item_emb, axis=1)
        all_feats = [final_interest, target_item_emb_sq] + other_features_emb
        deep_input = tf.concat(all_feats, axis=-1)
        
        x = self.bn(deep_input)
        x = self.dense1(x)
        x = self.dense2(x)
        logits = self.final_dense(x)
        output = keras.activations.sigmoid(logits)
        return output

# ================= 训练与保存 =================

input_spec = {}
for key in features_dict.keys():
    input_spec[key] = tf.TensorSpec(shape=[None, ], dtype=features_dict.get(key)[1], name=key)

model = DIEN(features_dict, SEQUENCE_PAIRS)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
auc_metric = tf.keras.metrics.AUC(name='auc')

@tf.function
def serving_function(data):
    result = {'prob': model(data)}
    return result

# 确保在 compile 前生成 concrete function，验证 tracing
output_func = serving_function.get_concrete_function(input_spec)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=[auc_metric])

print(f"Start Training DIEN (Seq Len: {MAX_SEQ_LEN})...")
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