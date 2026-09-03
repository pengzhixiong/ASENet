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