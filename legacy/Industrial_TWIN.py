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
raw_features_dict['imp_rank'] = (10, tf.string, 8, 0, False, 'iorder 第几次曝光')
raw_features_dict['buyer_sex'] = (10, tf.string, 8, 0, True, 'buyer_sex 性别')
raw_features_dict['visit_chnl1'] = (50, tf.string, 8, 0, True, 'visit_chnl1 注册一级渠道')
raw_features_dict['visit_chnl2'] = (500, tf.string, 8, 0, True, 'visit_chnl2 注册二级渠道')
raw_features_dict['country_id'] = (500, tf.string, 8, 0, True, 'countryId 国家')
raw_features_dict['site'] = (10, tf.string, 8, 0, True, 'site 三端类型')
raw_features_dict['lang'] = (50, tf.string, 8, 0, True, 'lang 语言')
raw_features_dict['hour'] = (72, tf.string, 8, 0, True, 'hour 小时')
raw_features_dict['day_of_week'] = (35, tf.string, 8, 0, True, 'weekDay 周几')
raw_features_dict['is_weekend'] = (5, tf.string, 8, 0, True, 'isWeekend 是否周末')
raw_features_dict['searchword'] = (1000000, tf.string, 8, 0, True, 'SearchWord 搜索词')
raw_features_dict['categorypred'] = (30000, tf.string, 8, 0, True, 'CategoryPredict 类目预测(知识图谱)接口返回的类目')
raw_features_dict['visit_days_num_30d'] = (100, tf.string, 8, 0, True, 'active_days_30d 活跃天数（最近30天)')
raw_features_dict['visit_prod_num_30d'] = (100, tf.string, 8, 0, True, 'visit_item_num_30d 用户最近30天浏览商品数')
raw_features_dict['addfav_prod_num_30d'] = (100, tf.string, 8, 0, True, 'favorite_item_num_30d 用户最近30天收藏商品数')
raw_features_dict['addcart_prod_num_30d'] = (100, tf.string, 8, 0, True, 'cart_item_num_30d 用户最近30天加购商品数')
raw_features_dict['last_confirm_interval_etl_days'] = (100, tf.string, 8, 0, True, 'recency 最后一次购买距离现在的天数')
raw_features_dict['confirm_days_frequency_his'] = (100, tf.string, 8, 0, True, 'frequency 购买频率(周期)')
raw_features_dict['started_rfx_num_365d'] = (100, tf.string, 8, 0, True, 'total_order_cnt 用户累计订单量')
raw_features_dict['started_rfx_num_30d'] = (100, tf.string, 8, 0, True, 'started_item_num_30d 用户最近30天下单商品数')
raw_features_dict['visit_cate1_pub_max_30d'] = (300, tf.string, 8, 0, True, 'Visit_cate1_30d 用户最近30天浏览主要一级类目')
raw_features_dict['addcart_cate1_pub_max_30d'] = (300, tf.string, 8, 0, True, 'Cart_cate1_30d 用户最近30天加购主要一级类目')
raw_features_dict['confirm_cate1_pub_num_30d'] = (300, tf.string, 8, 0, True, 'Confirmorder_cate1_30d 用户最近30天付款确认主要一级类目')
raw_features_dict['itemcode'] = (1000000, tf.string, 8, 0, True, 'item_code 商品id')
raw_features_dict['seller_id'] = (20000, tf.string, 8, 0, True, 'seller_id 卖家id')
raw_features_dict['cate_disp_id'] = (5000, tf.string, 8, 0, True, 'cate_disp_id 展示类目id')
raw_features_dict['is_free_ship'] = (5, tf.string, 8, 0, True, 'is_free_ship 是否包邮')
raw_features_dict['is_newuseronly'] = (5, tf.string, 8, 0, True, 'is_newuseronly 是否是正在参加新人专享价商品(1新人折扣)')
raw_features_dict['expo_1d'] = (100, tf.string, 8, 0, True, 'expo_1d 商品1天内曝光')
raw_features_dict['expo_7d'] = (200, tf.string, 8, 0, True, 'expo_7d 商品7天内曝光')
raw_features_dict['expo_30d'] = (300, tf.string, 8, 0, True, 'expo_30d 商品30天曝光数')
raw_features_dict['click_1d'] = (100, tf.string, 8, 0, True, 'click_1d 商品1天内点击')
raw_features_dict['click_7d'] = (100, tf.string, 8, 0, True, 'click_7d 商品7天内点击')
raw_features_dict['click_30d'] = (100, tf.string, 8, 0, True, 'click_30d 商品30天内点击')
raw_features_dict['favorite_30d'] = (100, tf.string, 8, 0, True, 'favorite_30d 商品收藏次数_30d ')
raw_features_dict['addcart_30d'] = (100, tf.string, 8, 0, True, 'addcart_30d 商品加购次数_30d ')
raw_features_dict['order_30d'] = (100, tf.string, 8, 0, True, 'order_30d 商品累计出单总数30d ')
raw_features_dict['order_all'] = (100, tf.string, 8, 0, True, 'order_all 商品总成单数 ')
raw_features_dict['review_all'] = (100, tf.string, 8, 0, True, 'review_all 商品历史总评论数 ')
raw_features_dict['search_expo_list_30d'] = (800000, tf.string, 8, 1, False, '30天曝光的广告ID序列')
raw_features_dict['search_click_list_30d'] = (800000, tf.string, 8, 1, False, '30天点击的商品ID序列')
raw_features_dict['search_cate_disp_list_30d'] = (800000, tf.string, 8, 1, True, '30天曝光的广告ID序列')
raw_features_dict['search_seller_list_30d'] = (800000, tf.string, 8, 1, True, '30天点击的商品ID序列')
raw_features_dict['search_query_list_30d'] = (800000, tf.string, 8, 1, False, '30天点击的商品ID序列')
raw_features_dict['is_click'] = (100, tf.float32, 8, 0, False, 'is_click 是否点击 ')

# --- 3. 数据加载配置 ---
label_col = 'is_click'

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

_SHUFFLE_SIZE = 10000
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


# =========================================================================
# [核心改造]: TWIN 论文提出的 Efficient Multi-Head Target Attention 层
# =========================================================================
class EfficientMHTA(tf.keras.layers.Layer):
    """
    对应论文中的 Target Attention in TWIN：
    处理提取的目标项固有特征(Inherent Features Kh)和序列特征。
    """

    def __init__(self, num_heads=2, key_dim=8, **kwargs):
        super(EfficientMHTA, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.output_dim = num_heads * key_dim

    def build(self, input_shape):
        # 对应论文中的 W^q, W^h, W^v, W^o 投影矩阵
        self.W_q = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='W_q')
        self.W_h = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='W_h')
        self.W_v = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='W_v')
        self.W_o = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='W_o')
        super(EfficientMHTA, self).build(input_shape)

    def call(self, target, seq, mask=None):
        # target: [Batch, Dim], seq: [Batch, Seq_len, Dim]
        batch_size = tf.shape(target)[0]
        seq_len = tf.shape(seq)[1]

        target = tf.expand_dims(target, axis=1)  # [Batch, 1, Dim]

        # 线性投影 (TWIN中 W_h(seq) 在线上服务时被物化缓存以突破算力瓶颈)
        q = self.W_q(target)
        k = self.W_h(seq)
        v = self.W_v(seq)

        # 调整多头注意力的形状: [Batch, Num_Heads, Seq_len/1, Key_Dim]
        q = tf.transpose(tf.reshape(q, [batch_size, 1, self.num_heads, self.key_dim]), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, [batch_size, seq_len, self.num_heads, self.key_dim]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, [batch_size, seq_len, self.num_heads, self.key_dim]), [0, 2, 1, 3])

        # 公式 6 核心: 打分 alpha = (K_h W^h)(q W^q)^T / sqrt(d_k)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))

        # 屏蔽填充词(Padding)的注意力得分
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)  # [Batch, 1, 1, Seq_len]
            padding_mask = (1.0 - mask) * -1e9
            scores += padding_mask

        # Softmax 并加权求和
        weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(weights, v)  # [Batch, Num_Heads, 1, Key_dim]

        # 还原输出并做最后的输出投影 W_o
        output = tf.reshape(tf.transpose(output, [0, 2, 1, 3]), [batch_size, 1, self.output_dim])
        output = tf.squeeze(output, axis=1)
        return self.W_o(output)


class CtrHashNN(keras.Model):
    def __init__(self, features_dict, **kwargs):
        super(CtrHashNN, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)

        # 实例化 TWIN 注意力模块 (因为我们拼合了cate(8维)和brand(8维)，所以设多头)
        self.twin_mhta = EfficientMHTA(num_heads=2, key_dim=8)

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
                result[f] = keras.layers.Embedding(input_dim=features_dict[f][0], output_dim=features_dict[f][2],
                                                   name=f + '_embedding')
        return result

    @tf.function
    def _format_inputs(self, inputs):
        features_recons = {}
        for key in self._features_dict.keys():
            if self._features_dict[key][3] == 1:
                # 针对变长序列特征：转换为稠密Tensor并用空字符填充对齐形状
                ragged = tf.strings.split(inputs.get(key), sep='^')
                dense = ragged.to_tensor(default_value="")
                features_recons[key] = dense
            else:
                features_recons[key] = inputs.get(key)
        return features_recons

    @tf.function
    def _preprocess_features(self, formatted_inputs):
        emb_dict = {}

        # 1. 统一提取并编码所有特征
        for key in self._features_dict.keys():
            val = formatted_inputs.get(key)
            if self._features_dict[key][1] == tf.string:
                x_int = self._hash_dict.get(key)(val)
                x_emb = self._emb_dict.get(key)(x_int)
                emb_dict[key] = x_emb
            else:
                # 针对 float 类型(如price)扩展最后维度，以确保拼接时不报错 [Batch, 1]
                emb_dict[key] = tf.expand_dims(tf.cast(val, tf.float32), axis=-1)

        # 2. 【TWIN 特征重组】构造 Target特征 和 历史序列特征
        # 固有目标特征 q
        target_emb = tf.keras.layers.concatenate([emb_dict['seller_id'], emb_dict['cate_disp_id']], axis=-1)

        # 序列特征 K_h
        seq_cate = emb_dict['search_cate_disp_list_30d']
        seq_brand = emb_dict['search_seller_list_30d']
        # 为了应对脏数据导致多列变长序列长度不一致，做长度裁剪对齐
        seq_len = tf.minimum(tf.shape(seq_cate)[1], tf.shape(seq_brand)[1])
        seq_cate = seq_cate[:, :seq_len, :]
        seq_brand = seq_brand[:, :seq_len, :]
        seq_emb = tf.keras.layers.concatenate([seq_cate, seq_brand], axis=-1)

        # 3. 生成 Mask 屏蔽空白词防止扰乱注意力分布
        mask_str = formatted_inputs.get('search_cate_disp_list_30d')[:, :seq_len]
        mask = tf.math.not_equal(mask_str, "")

        # 4. 【TWIN 核心计算】提取用户精准历史兴趣
        user_interest = self.twin_mhta(target_emb, seq_emb, mask)

        # 5. 组合其余 Scalar 特征进入深度网络
        final_inputs = [user_interest]
        for key in self._features_dict.keys():
            if self._features_dict[key][3] == 0:  # 仅放入单值特征
                final_inputs.append(emb_dict[key])

        processed_input = tf.keras.layers.concatenate(final_inputs, axis=-1)
        return processed_input

    @tf.function
    def call(self, data):
        data = self._format_inputs(data)
        x = self._preprocess_features(data)
        y_pred = self._dense(x)
        y_pred = keras.activations.sigmoid(y_pred)
        return y_pred


input_spec = {}
for key in features_dict.keys():
    input_spec[key] = tf.TensorSpec(shape=[None, ], dtype=features_dict.get(key)[1], name=key)

model = CtrHashNN(features_dict)
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
