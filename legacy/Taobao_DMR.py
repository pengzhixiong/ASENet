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

class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        # 位置编码矩阵，可训练
        self.position_embeddings = self.add_weight(
            shape=(sequence_length, output_dim),
            initializer="uniform",
            name="position_embeddings",
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length, embedding_dim)
        # 获取输入序列的实际长度
        length = tf.shape(inputs)[1]
        # 截取与输入序列等长的位置编码
        position_embeddings = self.position_embeddings[:length, :]
        # 将位置编码加到输入上
        return inputs + position_embeddings
        
# --- 模型修改开始 ---
class DMR(keras.Model):
    def __init__(self, features_dict, max_seq_len=50, **kwargs):
        super(DMR, self).__init__(**kwargs)

        self._features_dict = features_dict
        self.max_seq_len = max_seq_len

        # 1. 定义特征角色
        self.sequence_features = ['cate_his', 'brand_his']
        self.target_item_features = {'cate_his': 'cate_id', 'brand_his': 'brand'}
        self.other_sparse_features = [f for f in features_dict.keys()
                                      if f not in self.sequence_features and f not in self.target_item_features.values()
                                      and features_dict[f][1] == tf.string]
        self.dense_features = [f for f in features_dict.keys() if features_dict[f][1] == tf.float32]


        # 2. 初始化 Hashing 和 Embedding 层
        self._hash_dict = self._init_hash_dict(features_dict)
        self._emb_dict = self._init_emb_dict(features_dict)

        # ***新增***: 为每个序列特征创建位置编码层
        self.cate_pos_emb = PositionalEmbedding(self.max_seq_len, features_dict['cate_his'][2])
        self.brand_pos_emb = PositionalEmbedding(self.max_seq_len, features_dict['brand_his'][2])
        self.pos_emb_layers = {'cate_his': self.cate_pos_emb, 'brand_his': self.brand_pos_emb}

        # 3. 初始化 U2I 网络的 Attention 层
        # 使用 use_scale=True 可以让 Attention 更好地工作
        self.cate_attention = keras.layers.Attention(use_scale=True, name='cate_attention')
        self.brand_attention = keras.layers.Attention(use_scale=True, name='brand_attention')
        self.attention_layers = {'cate_his': self.cate_attention, 'brand_his': self.brand_attention}

        # 4. 初始化 Rank 网络的 MLP
        self.rank_net = tf.keras.Sequential([
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
                # ***关键修改2***: 对序列特征的Embedding层启用masking
                if f in self.sequence_features:
                    result[f] = keras.layers.Embedding(input_dim=features_dict[f][0], output_dim=features_dict[f][2],
                                                       name=f + '_embedding', mask_zero=True)
                else:
                    result[f] = keras.layers.Embedding(input_dim=features_dict[f][0], output_dim=features_dict[f][2],
                                                       name=f + '_embedding')
        return result
    
    @tf.function
    def _format_inputs(self, inputs):
        features_recons = {}
        for key in self._features_dict.keys():
            if self._features_dict[key][3] == 1:
                split_seq = tf.strings.split(tf.reshape(inputs.get(key), [-1]), sep='^')
                truncated_seq = split_seq[:, :self.max_seq_len]
                # ***关键修改1***: 将 RaggedTensor 转换为带有 padding 的 Dense Tensor
                # 空字符串""将被 Hashing 层哈希到 0，Embedding 层中 mask_zero=True 会处理这个 0
                dense_seq = truncated_seq.to_tensor(default_value="")
                features_recons[key] = dense_seq
            else:
                features_recons[key] = tf.reshape(inputs.get(key), [-1])
        return features_recons
    
    @tf.function
    def call(self, data):
        formatted_inputs = self._format_inputs(data)
        
        # 将所有需要embedding的特征都先计算好
        feature_embeddings = {}
        all_sparse_features = self.sequence_features + list(self.target_item_features.values()) + self.other_sparse_features
        for f in all_sparse_features:
            hashed_val = self._hash_dict[f](formatted_inputs[f])
            feature_embeddings[f] = self._emb_dict[f](hashed_val)

        # === Stage 1: U2I (User-to-Item) Network ===
        attention_outputs = []
        for seq_feat in self.sequence_features:
            target_feat = self.target_item_features[seq_feat]

            # Query: 候选物品 Embedding, shape [batch_size, 1, emb_dim]
            query = tf.expand_dims(feature_embeddings[target_feat], axis=1)
            # Key & Value: 历史行为序列 Embedding, shape [batch_size, seq_len, emb_dim]
            hist_seq_emb = feature_embeddings[seq_feat]
            # ***修改***: 在输入 Attention 层之前，加入位置编码
            hist_seq_emb_with_pos = self.pos_emb_layers[seq_feat](hist_seq_emb)
            
            # 使用带有位置信息的新Embedding进行Attention计算
            attention_output = self.attention_layers[seq_feat]([query, hist_seq_emb_with_pos])
            attention_output = tf.squeeze(attention_output, axis=1)
            attention_outputs.append(attention_output)

        # === Stage 2: Rank Network ===
        rank_inputs = []
        # 1. U2I 网络的输出
        rank_inputs.extend(attention_outputs)

        # 2. Target Item 特征本身也作为Rank网络的输入
        for target_feat in self.target_item_features.values():
            rank_inputs.append(feature_embeddings[target_feat])
        
        # 3. 其他离散特征
        for f in self.other_sparse_features:
            rank_inputs.append(feature_embeddings[f])
        
        # 4. 数值型特征
        for f in self.dense_features:
            if f == 'price':
                # price 特征的离散化和嵌入逻辑
                price_scaled = formatted_inputs.get(f) * 100.0
                price_binned = tf.cast(price_scaled, dtype=tf.int32)
                price_str = tf.strings.as_string(price_binned)
                hashed_feature = self._hash_dict.get(f)(price_str)
                embedded_feature = self._emb_dict.get(f)(hashed_feature)
                rank_inputs.append(embedded_feature)
            else:
                dense_input = tf.expand_dims(tf.cast(formatted_inputs[f], dtype=tf.float32), axis=1)
                rank_inputs.append(dense_input)
        
        # 拼接所有特征
        concatenated_input = tf.keras.layers.concatenate(rank_inputs, axis=-1)

        # 输入到 MLP
        y_pred = self.rank_net(concatenated_input)
        y_pred = keras.activations.sigmoid(y_pred)
        return y_pred
# --- 模型修改结束 ---



# 实例化新的 DMR 模型
model = DMR(features_dict)
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