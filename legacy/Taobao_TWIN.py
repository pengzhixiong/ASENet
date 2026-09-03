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

# --- 3. 数据加载配置 ---
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
        target_emb = tf.keras.layers.concatenate([emb_dict['cate_id'], emb_dict['brand']], axis=-1)

        # 序列特征 K_h
        seq_cate = emb_dict['cate_his']
        seq_brand = emb_dict['brand_his']
        # 为了应对脏数据导致多列变长序列长度不一致，做长度裁剪对齐
        seq_len = tf.minimum(tf.shape(seq_cate)[1], tf.shape(seq_brand)[1])
        seq_cate = seq_cate[:, :seq_len, :]
        seq_brand = seq_brand[:, :seq_len, :]
        seq_emb = tf.keras.layers.concatenate([seq_cate, seq_brand], axis=-1)

        # 3. 生成 Mask 屏蔽空白词防止扰乱注意力分布
        mask_str = formatted_inputs.get('cate_his')[:, :seq_len]
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
labels = [int(line.strip().split(',')[0]) for line in read]

if len(labels) > 0 and len(pre_cvr_list) > 0:
    test_losloss = log_loss(labels, pre_cvr_list)
    test_auc = roc_auc_score(labels, pre_cvr_list)
    test_pcoc = sum(pre_cvr_list)[0] / sum(labels) - 1 if sum(labels) > 0 else 0
    print("test_losloss:{:.6f}, test_auc:{:.6f}, test_pcoc:{:.6f}".format(test_losloss, test_auc, test_pcoc))
else:
    print("Test data is empty, skipping evaluation.")

final_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('final time: ', final_time)
