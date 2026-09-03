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
    
# 跟后面兼容,features_dict只保存有用的列
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


class AttentionLayer(keras.layers.Layer):
    """
    DIN模型的注意力激活单元
    """
    def __init__(self, hidden_units, activation='relu', **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
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
            paddings = tf.ones_like(attention_scores) * (-2**32 + 1)
            attention_scores = tf.where(mask, attention_scores, paddings)

        attention_weights = tf.nn.softmax(attention_scores)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        
        output = tf.reduce_sum(keys * attention_weights, axis=1)
        return output

class DIN(keras.Model):
    def __init__(self, features_dict, seq_len=50, **kwargs):
        super(DIN, self).__init__(**kwargs)
        self._features_dict = features_dict
        self.seq_len = seq_len
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)
        
        self.attention_cate = AttentionLayer(hidden_units=[64, 32], name='attention_cate')
        self.attention_brand = AttentionLayer(hidden_units=[64, 32], name='attention_brand')

        self._dense = tf.keras.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, name='dense_1', activation='relu'),
            keras.layers.Dense(64, name='dense_2', activation='relu'),
            keras.layers.Dense(1, name='dense_out')
        ])

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
                result[f] = keras.layers.Embedding(input_dim=features_dict[f][0], output_dim=features_dict[f][2], name=f + '_embedding')
        return result

    @tf.function
    def call(self, data):
        # 1. 处理序列特征：分割、填充/截断、哈希、嵌入和创建Mask
        # 处理'cate_his'
        cate_his_ragged = tf.strings.split(tf.reshape(data['cate_his'], [-1]), sep='^')
        cate_his_padded_str = cate_his_ragged.to_tensor(default_value='', shape=[None, self.seq_len])
        cate_his_mask = tf.not_equal(cate_his_padded_str, '')
        cate_his_hashed = self._hash_dict['cate_his'](cate_his_padded_str)
        cate_his_emb = self._emb_dict['cate_his'](cate_his_hashed)

        # 处理'brand_his'
        brand_his_ragged = tf.strings.split(tf.reshape(data['brand_his'], [-1]), sep='^')
        brand_his_padded_str = brand_his_ragged.to_tensor(default_value='', shape=[None, self.seq_len])
        brand_his_mask = tf.not_equal(brand_his_padded_str, '')
        brand_his_hashed = self._hash_dict['brand_his'](brand_his_padded_str)
        # ------------------- 关键修复点在这里！-------------------
        # 之前错误地使用了 brand_his_padded_str，现在修正为使用哈希后的 brand_his_hashed
        brand_his_emb = self._emb_dict['brand_his'](brand_his_hashed)
        # ---------------------------------------------------------

        # 2. 获取目标广告（query）的Embedding
        target_cate_emb = self._emb_dict['cate_id'](self._hash_dict['cate_id'](data['cate_id']))
        target_brand_emb = self._emb_dict['brand'](self._hash_dict['brand'](data['brand']))

        # 3. 通过AttentionLayer计算用户兴趣表示
        user_interest_cate = self.attention_cate([target_cate_emb, cate_his_emb], mask=cate_his_mask)
        user_interest_brand = self.attention_brand([target_brand_emb, brand_his_emb], mask=brand_his_mask)

        # 4. 准备其它特征
        other_features_emb = []
        for key in self._features_dict.keys():
            if key in ['cate_his', 'brand_his', 'cate_id', 'brand']:
                continue
            
            if self._features_dict[key][1] == tf.string:
                x_hashed = self._hash_dict.get(key)(data.get(key))
                x_emb = self._emb_dict.get(key)(x_hashed)
                other_features_emb.append(x_emb)
            elif key == 'price' and self._features_dict[key][1] == tf.float32:
                # price 特征的离散化和嵌入逻辑
                price_scaled = data.get(key) * 100.0
                price_binned = tf.cast(price_scaled, dtype=tf.int32)
                price_str = tf.strings.as_string(price_binned)
                hashed_feature = self._hash_dict.get(key)(price_str)
                embedded_feature = self._emb_dict.get(key)(hashed_feature)
                other_features_emb.append(embedded_feature)
            elif self._features_dict[key][1] == tf.float32:
                # 直接将数值特征reshape为 (batch_size, 1) 以便拼接
                reshaped_feature = tf.reshape(data.get(key), (-1, 1))
                other_features_emb.append(reshaped_feature)

        # 5. 拼接所有特征向量
        concat_input = tf.keras.layers.concatenate([
            target_cate_emb,
            target_brand_emb,
            user_interest_cate,
            user_interest_brand
        ] + other_features_emb, axis=-1)

        # 6. 通过MLP进行预测
        y_pred = self._dense(concat_input)
        y_pred = keras.activations.sigmoid(y_pred)
        return y_pred


model = DIN(features_dict)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
auc = tf.keras.metrics.AUC(name='my_auc')

model.compile(optimizer=optimizer, loss=loss, metrics=[auc])
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
