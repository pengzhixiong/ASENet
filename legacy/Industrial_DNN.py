# -*- coding: UTF-8 -*-
from sklearn.metrics import roc_auc_score, log_loss
import tensorflow as tf
import numpy as np
import os
from collections import OrderedDict
import tensorflow.keras as keras
from datetime import date, timedelta
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np.random.seed(42)
tf.random.set_seed(42)


raw_features_dict = OrderedDict()
# 特征们的字典，存储格式为(最大编码数目，输入特征类型，嵌入维度，特征类型（0为单值，1为多值），是否在本模型中选用，特征含义描述)
raw_features_dict['imp_rank'] = (10, tf.string, 8, 0, False, '第几次曝光')
raw_features_dict['buyer_sex'] = (10, tf.string, 8, 0, True, '性别')
raw_features_dict['visit_chnl1'] = (50, tf.string, 8, 0, True, '注册一级渠道')
raw_features_dict['visit_chnl2'] = (500, tf.string, 8, 0, True, '注册二级渠道')
raw_features_dict['country_id'] = (500, tf.string, 8, 0, True, '国家')
raw_features_dict['site'] = (10, tf.string, 8, 0, True, '三端类型')
raw_features_dict['lang'] = (50, tf.string, 8, 0, True, '语言')
raw_features_dict['hour'] = (72, tf.string, 8, 0, True, '小时')
raw_features_dict['day_of_week'] = (35, tf.string, 8, 0, True, '周几')
raw_features_dict['is_weekend'] = (5, tf.string, 8, 0, True, '是否周末')
raw_features_dict['searchword'] = (1000000, tf.string, 8, 0, True, '搜索词')
raw_features_dict['categorypred'] = (30000, tf.string, 8, 0, True, '类目预测(知识图谱)接口返回的类目')
raw_features_dict['visit_days_num_30d'] = (100, tf.string, 8, 0, True, '活跃天数（最近30天)')
raw_features_dict['visit_prod_num_30d'] = (100, tf.string, 8, 0, True, '用户最近30天浏览商品数')
raw_features_dict['addfav_prod_num_30d'] = (100, tf.string, 8, 0, True, '用户最近30天收藏商品数')
raw_features_dict['addcart_prod_num_30d'] = (100, tf.string, 8, 0, True, '用户最近30天加购商品数')
raw_features_dict['last_confirm_interval_etl_days'] = (100, tf.string, 8, 0, True, '最后一次购买距离现在的天数')
raw_features_dict['confirm_days_frequency_his'] = (100, tf.string, 8, 0, True, '购买频率(周期)')
raw_features_dict['started_rfx_num_365d'] = (100, tf.string, 8, 0, True, '用户累计订单量')
raw_features_dict['started_rfx_num_30d'] = (100, tf.string, 8, 0, True, '用户最近30天下单商品数')
raw_features_dict['visit_cate1_pub_max_30d'] = (300, tf.string, 8, 0, True, '用户最近30天浏览主要一级类目')
raw_features_dict['addcart_cate1_pub_max_30d'] = (300, tf.string, 8, 0, True, '用户最近30天加购主要一级类目')
raw_features_dict['confirm_cate1_pub_num_30d'] = (300, tf.string, 8, 0, True, '用户最近30天付款确认主要一级类目')
raw_features_dict['itemcode'] = (1000000, tf.string, 8, 0, True, '广告id')
raw_features_dict['seller_id'] = (20000, tf.string, 8, 0, True, '卖家id')
raw_features_dict['cate_disp_id'] = (5000, tf.string, 8, 0, True, '展示类目id')
raw_features_dict['is_free_ship'] = (5, tf.string, 8, 0, True, '是否包邮')
raw_features_dict['is_newuseronly'] = (5, tf.string, 8, 0, True, '是否是正在参加新人专享价商品(1新人折扣)')
raw_features_dict['expo_1d'] = (100, tf.string, 8, 0, True, '商品1天内曝光')
raw_features_dict['expo_7d'] = (200, tf.string, 8, 0, True, '商品7天内曝光')
raw_features_dict['expo_30d'] = (300, tf.string, 8, 0, True, '商品30天曝光数')
raw_features_dict['click_1d'] = (100, tf.string, 8, 0, True, '商品1天内点击')
raw_features_dict['click_7d'] = (100, tf.string, 8, 0, True, '商品7天内点击')
raw_features_dict['click_30d'] = (100, tf.string, 8, 0, True, '商品30天内点击')
raw_features_dict['favorite_30d'] = (100, tf.string, 8, 0, True, '商品收藏次数_30d ')
raw_features_dict['addcart_30d'] = (100, tf.string, 8, 0, True, '商品加购次数_30d ')
raw_features_dict['order_30d'] = (100, tf.string, 8, 0, True, '商品累计出单总数30d ')
raw_features_dict['order_all'] = (100, tf.string, 8, 0, True, '商品总成单数 ')
raw_features_dict['review_all'] = (100, tf.string, 8, 0, True, '商品历史总评论数 ')
raw_features_dict['search_expo_list_30d'] = (800000, tf.string, 8, 1, False, '30天曝光的广告id序列')
raw_features_dict['search_click_list_30d'] = (800000, tf.string, 8, 1, False, '30天点击的广告id序列')
raw_features_dict['search_cate_disp_list_30d'] = (800000, tf.string, 8, 1, False, '30天点击的展示类目id序列')
raw_features_dict['search_seller_list_30d'] = (800000, tf.string, 8, 1, False, '30天点击的卖家id序列')
raw_features_dict['search_query_list_30d'] = (800000, tf.string, 8, 1, False, '30天点击的搜索词序列')
raw_features_dict['is_click'] = (100, tf.float32, 8, 0, False, '是否点击 label')

label_col = 'is_click'

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


train_files = '/data2/jupyter/nobackup/dataset/searchv6/train_20250924_ns.csv'
valid_files = '/data2/jupyter/nobackup/dataset/searchv6/valid_20250924_ns.csv'
test_files = '/data2/jupyter/nobackup/dataset/searchv6/fmsearch_20250924_shuf.csv'
serving_model_path = '/data2/jupyter/nobackup/model/search_ctr_gan_v1/'


BATCH_SIZE = 2048
EPOCHS = 1
LEARNING_RATE = 0.001

train_dataset = get_dataset(train_files, shuffle=True, num_epochs=1, batch_size=BATCH_SIZE)
valid_dataset = get_dataset(valid_files, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)
test_dataset = get_dataset(test_files, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)


class PengHashNN(keras.Model):
    def __init__(self, features_dict, **kwargs):
        super(PengHashNN, self).__init__(**kwargs)

        self._features_dict = features_dict
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)
        self._dense = tf.keras.Sequential([keras.layers.Dense(128, name='dense_1', activation='relu'),
                                           keras.layers.Dense(64, name='dense_2', activation='relu'),
                                           keras.layers.Dense(1, name='dense_out')])

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
                features_recons[key] = tf.strings.split(tf.reshape(inputs.get(key), [-1]), sep=',')
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
                processed_inputs.append(formatted_inputs.get(key))
        processed_input = tf.keras.layers.concatenate(processed_inputs, axis=-1)
        return processed_input

    @tf.function
    def call(self, data):
        data = self._format_inputs(data)
        x = self._preprocess_features(data)
        y_pred = self._dense(x)
        y_pred = keras.activations.sigmoid(y_pred)
        return y_pred


model = PengHashNN(features_dict)
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

