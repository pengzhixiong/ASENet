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

# 请确保路径存在
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
# HierDiffuse 核心模块定义
# =========================================================================

class TargetAttention(keras.layers.Layer):
    """经典的 Target Attention 机制，用于从序列中提取与目标 Item 相关的兴趣"""
    def __init__(self, **kwargs):
        super(TargetAttention, self).__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(1, activation=None)

    def call(self, target_emb, seq_emb, mask=None):
        # target_emb: [B, D], seq_emb: [B, T, D]
        target_emb_expanded = tf.expand_dims(target_emb, axis=1) # [B, 1, D]
        target_emb_tiled = tf.tile(target_emb_expanded, [1, tf.shape(seq_emb)[1], 1]) # [B, T, D]
        
        concat_info = tf.concat([target_emb_tiled, seq_emb, target_emb_tiled - seq_emb, target_emb_tiled * seq_emb], axis=-1)
        
        attention_score = self.dense2(self.dense1(concat_info)) # [B, T, 1]
        
        if mask is not None:
            paddings = tf.ones_like(attention_score) * (-2 ** 32 + 1)
            mask_expanded = tf.expand_dims(mask, axis=-1)
            attention_score = tf.where(mask_expanded, attention_score, paddings)
            
        attention_weight = tf.nn.softmax(attention_score, axis=1) # [B, T, 1]
        output = tf.reduce_sum(seq_emb * attention_weight, axis=1) # [B, D]
        return output

class HierDiffuseFusion(keras.layers.Layer):
    """
    HierDiffuse 核心层：实现融合即降噪
    包含：SGD (语义引导解耦) 和 TCC (轨迹收敛约束的单步前向映射)
    """
    def __init__(self, dim, **kwargs):
        super(HierDiffuseFusion, self).__init__(**kwargs)
        self.dim = dim
        # SGD 语义解耦权重网络：计算短期序列各个行为的真实引导强度
        self.sgd_net = keras.layers.Dense(1, activation='sigmoid', name="sgd_weight")
        
        # TCC 单步映射网络：模拟扩散模型的反向去噪过程
        # 使用类似 FiLM (Feature-wise Linear Modulation) 的结构注入 Short-term Condition
        self.tcc_dense1 = keras.layers.Dense(dim * 2, activation='relu')
        self.tcc_dense2 = keras.layers.Dense(dim)
        
        self.gamma_dense = keras.layers.Dense(dim)
        self.beta_dense = keras.layers.Dense(dim)

    def call(self, long_rep, short_seq_emb, target_emb):
        """
        long_rep: [B, D] (从长序列提取的宏观偏好, 作为基础噪声先验)
        short_seq_emb: [B, T_s, D] (短期序列的具体 Embedding)
        target_emb: [B, D] (当前候选广告)
        """
        # 1. SGD (Semantic Guidance Disentanglement)
        # 用 target 和 short_seq 交互，过滤掉短期的误触/噪声行为
        target_expanded = tf.expand_dims(target_emb, axis=1) # [B, 1, D]
        target_tiled = tf.tile(target_expanded, [1, tf.shape(short_seq_emb)[1], 1])
        sgd_input = tf.concat([short_seq_emb, target_tiled], axis=-1)
        omega_i = self.sgd_net(sgd_input) # [B, T_s, 1] 适应性缩放因子
        
        # 加权聚合得到纯净的短期引导条件 (Condition)
        short_cond = tf.reduce_sum(short_seq_emb * omega_i, axis=1) # [B, D]
        
        # 2. TCC (Trajectory Convergence Constraint) 模拟单步去噪映射
        # 这里用 short_cond 去调制 long_rep (Long -> Short 扩散范式)
        gamma = self.gamma_dense(short_cond) # [B, D]
        beta = self.beta_dense(short_cond)   # [B, D]
        
        # 模拟一步 Denoising: x_0 = \Phi(x_t, condition)
        x_t = self.tcc_dense1(long_rep)
        x_t = self.tcc_dense2(x_t)
        
        # FiLM 融合 (条件引导)
        fusion_rep = gamma * x_t + beta
        
        # 残差连接保护原始信息
        return fusion_rep + long_rep

# =========================================================================
# 模型主体构建
# =========================================================================

class CtrHashNN(keras.Model):
    def __init__(self, features_dict, seq_max_len=20, short_seq_len=5, **kwargs):
        super(CtrHashNN, self).__init__(**kwargs)
        self._features_dict = features_dict
        self.seq_max_len = seq_max_len       # 截断序列最大长度
        self.short_seq_len = short_seq_len   # 定义最近的N个点击为短期兴趣
        
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)
        
        # 初始化 HierDiffuse 相关的网络层
        self.long_attention = TargetAttention()
        self.hier_diffuse_fusion = HierDiffuseFusion(dim=8) # 嵌入维度为8
        
        self._dense = tf.keras.Sequential([
            keras.layers.Dense(128, name='dense_1', activation='relu'),
            keras.layers.Dense(64, name='dense_2', activation='relu'),
            keras.layers.Dense(1, name='dense_out')
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
                result[f] = keras.layers.Embedding(input_dim=features_dict[f][0], output_dim=features_dict[f][2], name=f + '_embedding')
        return result

    @tf.function
    def _format_inputs(self, inputs):
        features_recons = {}
        for key in self._features_dict.keys():
            if self._features_dict[key][3] == 1:
                # 处理序列特征：RaggedTensor 转为 Dense Tensor，并固定长度，方便切分长短期
                ragged_seq = tf.strings.split(tf.reshape(inputs.get(key), [-1]), sep='^')
                dense_seq = ragged_seq.to_tensor(default_value="", shape=[None, self.seq_max_len])
                features_recons[key] = dense_seq
            else:
                features_recons[key] = tf.reshape(inputs.get(key), [-1])
        return features_recons

    @tf.function
    def _preprocess_features(self, formatted_inputs):
        other_features = []
        seq_features_emb = {}
        target_features_emb = {}
        
        # 第一步：提取 Embedding
        for key in self._features_dict.keys():
            if self._features_dict[key][1] == tf.string:
                x_int = self._hash_dict.get(key)(formatted_inputs.get(key))
                x_emb = self._emb_dict.get(key)(x_int)
                
                if self._features_dict[key][3] == 1:
                    seq_features_emb[key] = x_emb # [B, MAX_LEN, D]
                else:
                    other_features.append(x_emb) # [B, D]
                    # 识别 target 目标特征用于 Attention 和 SGD
                    if key in ['cate_disp_id', 'seller_id']:
                        target_features_emb[key] = x_emb
            else:
                other_features.append(tf.expand_dims(formatted_inputs.get(key), axis=-1))

        # 第二步：HierDiffuse 长短期兴趣切分与融合
        fusion_reps = []
        
        # 这里我们将 cate_his 与 cate_id 匹配，brand_his 与 brand 匹配
        seq_target_pairs = [('search_cate_disp_list_30d', 'cate_disp_id'), ('search_seller_list_30d', 'seller_id')]
        
        for seq_key, target_key in seq_target_pairs:
            if seq_key in seq_features_emb and target_key in target_features_emb:
                full_seq_emb = seq_features_emb[seq_key] # [B, 50, D]
                target_emb = target_features_emb[target_key] # [B, D]
                
                # 1. 物理切分：长短期序列
                long_seq_emb = full_seq_emb[:, :-self.short_seq_len, :]  # [B, 40, D]
                short_seq_emb = full_seq_emb[:, -self.short_seq_len:, :] # [B, 10, D]
                
                # 构建 Mask (过滤掉 padding 的空字符串算出的 embedding，假设 hash 0 为 padding)
                long_mask = tf.not_equal(tf.reduce_sum(tf.abs(long_seq_emb), axis=-1), 0.0)
                
                # 2. 长期特征 h_L：通过 Target Attention 提取宏观偏好
                h_long = self.long_attention(target_emb, long_seq_emb, mask=long_mask)
                
                # 3. 核心融合：使用 HierDiffuse 的 TCC 和 SGD
                # h_fusion = \Phi(h_long, short_seq_emb | target)
                h_fusion = self.hier_diffuse_fusion(h_long, short_seq_emb, target_emb)
                fusion_reps.append(h_fusion)

        # 第三步：将融合后的序列特征与其它单值特征进行拼接
        processed_inputs = other_features + fusion_reps
        processed_input = tf.keras.layers.concatenate(processed_inputs, axis=-1)
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