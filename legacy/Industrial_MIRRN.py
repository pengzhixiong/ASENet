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

# ==================== 特征定义 ====================
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
MAX_SEQ_LEN = 20  # 设定序列最大长度，供FFT对齐使用
LOCAL_SEQ_LEN = 5  # LASU 局部序列长度

train_dataset = get_dataset(train_files, shuffle=True, num_epochs=1, batch_size=BATCH_SIZE)
valid_dataset = get_dataset(valid_files, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)
test_dataset = get_dataset(test_files, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)


# ==================== 核心自定义层 (修复版) ====================

class TargetAttention(keras.layers.Layer):
    """ 模拟 TASU (Target-Aware Search Unit) 和 Target Attention """

    def __init__(self, **kwargs):
        super(TargetAttention, self).__init__(**kwargs)

    def call(self, query, keys):
        # query: [B, 1, E], keys: [B, S, E]
        query = tf.expand_dims(query, axis=1) if len(query.shape) == 2 else query
        attention_scores = tf.matmul(query, keys, transpose_b=True)  # [B, 1, S]
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        output = tf.matmul(attention_scores, keys)  # [B, 1, E]

        # 修复点2：必须将 axis=1 维度压缩掉，否则后面计算 seq_t 会引起维度广播灾难
        return tf.squeeze(output, axis=1), tf.squeeze(attention_scores, axis=1)


class MHFTLayer(keras.layers.Layer):
    """ 核心创新：Multi-Head Fourier Transformer (MHFT) """

    def __init__(self, seq_len, emb_dim, **kwargs):
        super(MHFTLayer, self).__init__(**kwargs)
        # 修复点1：不再依赖运行时可能为 None 的 input_shape，直接使用明确传入的长度
        self.S = seq_len
        self.E = emb_dim

    def build(self, input_shape):
        # 使用确定的 self.S 和 self.E 初始化频域权重
        self.W_real = self.add_weight(name='w_real', shape=(self.S, self.E), initializer="glorot_uniform",
                                      trainable=True)
        self.W_imag = self.add_weight(name='w_imag', shape=(self.S, self.E), initializer="glorot_uniform",
                                      trainable=True)
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        super(MHFTLayer, self).build(input_shape)

    def call(self, x):
        # 1. 傅里叶变换到频域 (沿序列维度)
        x_t = tf.transpose(x, [0, 2, 1])  # [B, E, S]
        x_complex = tf.cast(x_t, tf.complex64)
        x_freq = tf.signal.fft(x_complex)  # 复杂度 O(N log N)
        x_freq = tf.transpose(x_freq, [0, 2, 1])  # [B, S, E]

        # 2. 频域内的乘法融合交互信息
        W_complex = tf.complex(self.W_real, self.W_imag)
        freq_out = x_freq * W_complex

        # 3. 逆傅里叶变换回时域
        freq_out_t = tf.transpose(freq_out, [0, 2, 1])  # [B, E, S]
        x_out_complex = tf.signal.ifft(freq_out_t)
        x_out = tf.math.real(tf.transpose(x_out_complex, [0, 2, 1]))  # [B, S, E]

        # 4. 残差连接与 LayerNorm
        return self.layer_norm(x + x_out)


# ==================== MIRRN 主网络 (修复版) ====================

class MIRRN(keras.Model):
    def __init__(self, features_dict, **kwargs):
        super(MIRRN, self).__init__(**kwargs)
        self._features_dict = features_dict
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)

        # 修复点3：明确定义三个序列 [全局, 局部, 目标感知] 的长度
        self.seq_lengths = [MAX_SEQ_LEN, LOCAL_SEQ_LEN, MAX_SEQ_LEN]

        # BSRM 模块子层：将确定的长度传给底层
        self.tapes = [keras.layers.Embedding(input_dim=l, output_dim=16) for l in self.seq_lengths]
        self.mhft_layers = [MHFTLayer(seq_len=l, emb_dim=16) for l in self.seq_lengths]

        # MIAM & 预测层
        self.target_attention = TargetAttention()
        self.miam_attention = keras.layers.MultiHeadAttention(num_heads=1, key_dim=8)
        self._dense = tf.keras.Sequential([
            keras.layers.Dense(128, name='mlp_1', activation='relu'),
            keras.layers.Dense(64, name='mlp_2', activation='relu'),
            keras.layers.Dense(1, name='prediction_layer', activation='sigmoid')
        ])

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
                ragged = tf.strings.split(tf.reshape(inputs.get(key), [-1]), sep='^')
                tensor = ragged.to_tensor(default_value='', shape=[None, MAX_SEQ_LEN])
                features_recons[key] = tensor
            else:
                features_recons[key] = tf.reshape(inputs.get(key), [-1])
        return features_recons

    def _get_bsrm_output(self, seq_emb, i):
        """ Behavior Sequence Refinement Module """
        # 修复点4：抛弃 tf.shape 动态获取长度，直接使用静态已知的固定长度，保平安
        seq_len = self.seq_lengths[i]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_emb = self.tapes[i](positions)  # [S, E]

        # 1. Target-Aware Position Encoding (TAPE)
        seq_emb = seq_emb + tf.expand_dims(pos_emb, axis=0)  # [B, S, E] + [1, S, E]

        # 2. Multi-Head Fourier Transformer (MHFT)
        refined_seq = self.mhft_layers[i](seq_emb)

        # 3. Average Pooling 获取兴趣向量
        return tf.reduce_mean(refined_seq, axis=1)  # [B, E]

    @tf.function
    def call(self, data):
        data = self._format_inputs(data)

        dense_inputs = []
        target_embs = []
        hist_embs_cate, hist_embs_brand = None, None

        for key in self._features_dict.keys():
            if self._features_dict[key][1] == tf.string:
                x_int = self._hash_dict.get(key)(data.get(key))
                x_emb = self._emb_dict.get(key)(x_int)

                if key in ['cate_disp_id', 'seller_id']:
                    target_embs.append(x_emb)
                elif key == 'search_cate_disp_list_30d':
                    hist_embs_cate = x_emb
                elif key == 'search_seller_list_30d':
                    hist_embs_brand = x_emb
                elif self._features_dict[key][3] != 1:
                    dense_inputs.append(x_emb)
            else:
                dense_inputs.append(tf.expand_dims(data.get(key), -1))

        # 融合 Target 和 History
        target_item = tf.concat(target_embs, axis=-1)  # [B, 16]
        hist_seq = tf.concat([hist_embs_cate, hist_embs_brand], axis=-1)  # [B, 20, 16]

        # 2. MIRM: 构造三种粒度的子序列
        seq_g = hist_seq
        seq_l = hist_seq[:, :LOCAL_SEQ_LEN, :]

        # target_weights 现在维度是 [B, 20]，完美匹配后续计算
        _, target_weights = self.target_attention(target_item, hist_seq)
        seq_t = hist_seq * tf.expand_dims(target_weights, axis=-1)  # [B, 20, 16] * [B, 20, 1]

        # 3. BSRM
        E_g = self._get_bsrm_output(seq_g, 0)  # [B, 16]
        E_l = self._get_bsrm_output(seq_l, 1)  # [B, 16]
        E_t = self._get_bsrm_output(seq_t, 2)  # [B, 16]

        # 4. MIAM
        E_all = tf.stack([E_g, E_l, E_t], axis=1)  # [B, 3, 16]
        target_query = tf.expand_dims(target_item, axis=1)  # [B, 1, 16]

        E_u = self.miam_attention(query=target_query, value=E_all, key=E_all)
        E_u = tf.squeeze(E_u, axis=1)  # [B, 16]

        # 5. Prediction Layer
        context_features = tf.keras.layers.concatenate(dense_inputs, axis=-1)
        final_input = tf.keras.layers.concatenate([target_item, E_u, context_features], axis=-1)

        y_pred = self._dense(final_input)
        return y_pred


# ==================== 编译与训练流程 ====================
input_spec = {}
for key in features_dict.keys():
    input_spec[key] = tf.TensorSpec(shape=[None, ], dtype=features_dict.get(key)[1], name=key)

model = MIRRN(features_dict)
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

# ==================== 保存与评估 ====================
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
