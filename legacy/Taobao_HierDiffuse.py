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

# 请确保路径存在
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
    def __init__(self, features_dict, seq_max_len=50, short_seq_len=10, **kwargs):
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
                    if key in ['cate_id', 'brand']:
                        target_features_emb[key] = x_emb
            else:
                other_features.append(tf.expand_dims(formatted_inputs.get(key), axis=-1))

        # 第二步：HierDiffuse 长短期兴趣切分与融合
        fusion_reps = []
        
        # 这里我们将 cate_his 与 cate_id 匹配，brand_his 与 brand 匹配
        seq_target_pairs = [('cate_his', 'cate_id'), ('brand_his', 'brand')]
        
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