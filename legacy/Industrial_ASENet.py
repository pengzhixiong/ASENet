from sklearn.metrics import roc_auc_score, log_loss
import tensorflow as tf
import numpy as np
import os
from collections import OrderedDict
import tensorflow.keras as keras
from datetime import date, timedelta
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# --- 1. 全局配置与环境设置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np.random.seed(42)
tf.random.set_seed(42)

# --- 2. 特征定义 ---
# 特征们的字典，存储格式为(最大编码数目，输入特征类型，嵌入维度，特征类型（0为单值，1为多值序列），是否在本模型中选用，特征含义描述)
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


# --- 4. 文件路径与超参数 ---
train_files = '/data2/jupyter/nobackup/dataset/searchv6/train_20250924_ns.csv'
valid_files = '/data2/jupyter/nobackup/dataset/searchv6/valid_20250924_ns.csv'
test_files = '/data2/jupyter/nobackup/dataset/searchv6/fmsearch_20250924_shuf.csv'
serving_model_path = '/data2/jupyter/nobackup/model/search_ctr_gan_v1/'

# 超参数
BATCH_SIZE = 2048
EPOCHS = 1
LR_G = 0.001  # WGAN-GP 通常使用更小的学习率
LR_D = 0.001  # WGAN-GP 通常使用更小的学习率
CONTENT_LOSS_WEIGHT = 20.0  # 内容损失的权重
GP_WEIGHT = 10.0  # 梯度惩罚的权重，这是 WGAN-GP 的核心超参数
NUM_EXPERTS = 4  # 专家数量
EXPERT_HIDDEN_UNITS = [64, 32]  # 每个专家网络的隐藏层结构
MoE_BALANCE = 0.2


# ******************** 新增: MoE 层 (可复用) ********************
class MoELayer(keras.layers.Layer):
    """
    Mixture of Experts (MoE) Layer
    """

    def __init__(self, num_experts, expert_hidden_units, output_dim=1, **kwargs):
        super(MoELayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.output_dim = output_dim

        self.experts = []
        for i in range(self.num_experts):
            expert_layers = []
            for units in expert_hidden_units:
                expert_layers.append(keras.layers.Dense(units, activation='relu', name=f'expert_{i}_dense_{units}'))
            expert_layers.append(keras.layers.Dense(self.output_dim, name=f'expert_{i}_output'))
            self.experts.append(tf.keras.Sequential(expert_layers, name=f'expert_{i}'))

        self.gating_network = keras.layers.Dense(self.num_experts, activation='softmax', name='gating_network')

    def call(self, inputs):
        gate_weights = self.gating_network(inputs)
        
        mean_gate_weights = tf.reduce_mean(gate_weights, axis=0) 
        load_balance_loss = self.num_experts * tf.reduce_sum(tf.square(mean_gate_weights)) - 1.0
        self.add_loss(load_balance_loss) 
        
        expert_outputs = [expert(inputs) for expert in self.experts]
        stacked_expert_outputs = tf.stack(expert_outputs, axis=2)
        expanded_gate_weights = tf.expand_dims(gate_weights, axis=1)
        weighted_expert_outputs = stacked_expert_outputs * expanded_gate_weights
        final_output = tf.reduce_sum(weighted_expert_outputs, axis=2)
        return final_output


# --- 5. 模型定义 (WGAN-GP 架构) ---
class FeatureExtractor(keras.Model):

    def __init__(self, features_dict, **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs)
        self.features_dict = {k: v for k, v in features_dict.items() if v[4]}
        self.sequence_max_len = 20

        self.sharing_map = {
            'cate_disp_id': 'cate',
            'search_cate_disp_list_30d': 'cate',
            'seller_id': 'seller',
            'search_seller_list_30d': 'seller',
        }
        self.base_feature_map = {
            'cate': 'cate_disp_id',
            'seller': 'seller_id',
        }

        self.shared_hash_layers = {}
        self.shared_embedding_layers = {}
        self.hash_layers = {}
        self.embedding_layers = {}
        self.lstm_layers = {}

        for group_name, base_feature_name in self.base_feature_map.items():
            max_vocab, _, emb_dim, _, _, _ = self.features_dict[base_feature_name]
            self.shared_hash_layers[group_name] = keras.layers.Hashing(
                num_bins=max_vocab, name=f'shared_{group_name}_hash'
            )
            self.shared_embedding_layers[group_name] = keras.layers.Embedding(
                input_dim=max_vocab, output_dim=emb_dim,
                name=f'shared_{group_name}_embedding', mask_zero=True
            )

        for name, (max_vocab, dtype, emb_dim, f_type, _, _) in self.features_dict.items():
            if name not in self.sharing_map and dtype == tf.string:
                self.hash_layers[name] = keras.layers.Hashing(num_bins=max_vocab, name=f'{name}_hash')
                self.embedding_layers[name] = keras.layers.Embedding(
                    input_dim=max_vocab, output_dim=emb_dim, name=f'{name}_embedding', mask_zero=True
                )

            if f_type == 1:
                self.lstm_layers[name] = keras.layers.LSTM(
                    units=emb_dim, name=f'{name}_lstm', recurrent_activation='sigmoid'
                )

    def get_embedding_list(self, inputs):
        processed_features = []
        for name, (_, dtype, _, f_type, _, _) in self.features_dict.items():
            if dtype != tf.string: continue

            feature_tensor = inputs[name]
            hash_layer, embedding_layer = None, None

            if name in self.sharing_map:
                group_name = self.sharing_map[name]
                hash_layer = self.shared_hash_layers[group_name]
                embedding_layer = self.shared_embedding_layers[group_name]
            else:
                hash_layer = self.hash_layers[name]
                embedding_layer = self.embedding_layers[name]

            if f_type == 1:
                split_feature = tf.strings.split(feature_tensor, sep='^')
                split_feature = tf.ragged.boolean_mask(split_feature, tf.not_equal(split_feature, ""))
                truncated_feature = split_feature[:, :self.sequence_max_len]

                hashed_feature = hash_layer(truncated_feature)
                dense_hashed_feature = hashed_feature.to_tensor()

                # 获取当前 Batch 的真实序列长度
                seq_len = tf.shape(dense_hashed_feature)[1]
                # 计算需要补齐的长度：如果 seq_len=0，则 pad_len=1；如果 seq_len>0，则 pad_len=0
                pad_len = tf.maximum(1 - seq_len, 0)
                # 使用 tf.pad 在序列(时间步)维度后面补 0，确保 LSTM 至少执行 1 步
                dense_hashed_feature = tf.pad(dense_hashed_feature, [[0, 0], [0, pad_len]])
                embedded_feature = embedding_layer(dense_hashed_feature)

                lstm_output = self.lstm_layers[name](embedded_feature)
                processed_features.append(lstm_output)

            else:
                hashed_feature = hash_layer(feature_tensor)
                embedded_feature = embedding_layer(hashed_feature)
                processed_features.append(embedded_feature)

        return processed_features

    def call(self, inputs):
        """处理输入特征并拼接成一个扁平的向量。"""
        embedding_list = self.get_embedding_list(inputs)
        return keras.layers.concatenate(embedding_list, axis=-1)


class Generator(FeatureExtractor):
    """生成器：标准的CTR预估模型 (保持不变)"""

    def __init__(self, features_dict, **kwargs):
        super(Generator, self).__init__(features_dict, **kwargs)
        self.mlp = tf.keras.Sequential([
            keras.layers.Dense(128, activation='relu', name='g_dense_1'),
            keras.layers.Dense(64, activation='relu', name='g_dense_2'),
            keras.layers.Dense(1, activation='sigmoid', name='g_output')  # 输出仍是概率，用于内容损失
        ], name='GeneratorMLP')

    def call(self, inputs):
        feature_vector = super(Generator, self).call(inputs)
        return self.mlp(feature_vector)


# ******************** 修改: 判别器使用MoE结构 ********************
class DiscriminatorWithMoE(FeatureExtractor):
    """判别器：使用MoE结构判断 (特征, CTR) 对的真实性"""

    def __init__(self, features_dict, num_experts, expert_hidden_units, **kwargs):
        super(DiscriminatorWithMoE, self).__init__(features_dict, **kwargs)
        # 将原有的 MLP 替换为 MoE 层
        self.moe_layer = MoELayer(
            num_experts=num_experts,
            expert_hidden_units=expert_hidden_units,
            output_dim=1,  # 输出一个logit值，判断真伪
            name='Discriminator_MoE_Layer'
        )

    def call(self, inputs):
        features, ctr = inputs  # 输入是一个列表 [features_dict, ctr_tensor]
        feature_vector = super(DiscriminatorWithMoE, self).call(features)
        ctr_reshaped = tf.reshape(ctr, shape=(-1, 1))

        # 将特征向量与CTR值拼接，作为MoE网络的输入
        combined_input = keras.layers.concatenate([feature_vector, ctr_reshaped], axis=-1)

        # 通过MoE层得到logits
        logits = self.moe_layer(combined_input)

        # 应用最终的sigmoid激活函数得到概率
        return logits


class GAN(keras.Model):
    """GAN模型，封装了G和D以及自定义的训练逻辑"""

    def __init__(self, generator, discriminator, **kwargs):
        super(GAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.ctr_loss_tracker = keras.metrics.Mean(name="ctr_loss")  # 内容损失
        self.adv_loss_tracker = keras.metrics.Mean(name="adv_loss")  # 对抗损失
        self.auc_tracker = tf.keras.metrics.AUC(name='g_auc')  # 跟踪生成器的AUC
        self.gp_tracker = keras.metrics.Mean(name="gp")  # 新增：梯度惩罚项追踪

    def compile(self, optimizer_g, optimizer_d, loss_fn):
        super(GAN, self).compile()
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker, self.ctr_loss_tracker,
                self.adv_loss_tracker, self.gp_tracker, self.auc_tracker]

    def gradient_penalty(self, batch_size, real_ctrs, fake_ctrs, features):
        """ 计算梯度惩罚 """
        # 1. 在真实CTR和生成CTR之间进行随机插值
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        interpolated_ctrs = real_ctrs + alpha * (fake_ctrs - real_ctrs)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_ctrs)
            # 2. 计算插值点的批评家输出
            pred = self.discriminator([features, interpolated_ctrs], training=True)

        # 3. 计算批评家输出对插值点的梯度
        grads = gp_tape.gradient(pred, [interpolated_ctrs])[0]
        # 4. 计算梯度的L2范数
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        # 5. 计算梯度惩罚 (目标是使范数接近1)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, data):
        features, real_labels = data
        real_labels = tf.cast(real_labels, tf.float32)
        real_labels = tf.reshape(real_labels, (-1, 1))

        batch_size = tf.shape(real_labels)[0]

        valid = tf.ones((batch_size, 1)) * 0.9
        fake = tf.ones((batch_size, 1)) * 0.1

        # --- 训练判别器 ---
        with tf.GradientTape() as tape:
            gen_ctrs = self.generator(features, training=True)

            real_pred = self.discriminator([features, real_labels], training=True)
            fake_pred = self.discriminator([features, gen_ctrs], training=True)
            d_loss_real = self.loss_fn(valid, real_pred)

            d_loss_fake = self.loss_fn(fake, fake_pred)

            c_wasserstein_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
            # 梯度惩罚
            gp = self.gradient_penalty(batch_size, real_labels, gen_ctrs, features)
            moe_balance_loss = sum(self.discriminator.losses) 
            # 批评家总损失
            d_loss = c_wasserstein_loss + GP_WEIGHT * gp + MoE_BALANCE * moe_balance_loss

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.optimizer_d.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # --- 训练生成器 ---
        with tf.GradientTape() as tape:
            gen_ctrs = self.generator(features, training=True)
            validity = self.discriminator([features, gen_ctrs], training=True)

            loss_ctr = self.loss_fn(real_labels, gen_ctrs)
            # 对抗损失 (Wasserstein): 衡量欺骗批评家的能力
            loss_adv = -tf.reduce_mean(validity)  # 目标是最大化批评家分数

            g_loss = CONTENT_LOSS_WEIGHT * loss_ctr + loss_adv

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.optimizer_g.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.ctr_loss_tracker.update_state(loss_ctr)
        self.adv_loss_tracker.update_state(loss_adv)
        self.auc_tracker.update_state(real_labels, gen_ctrs)
        self.gp_tracker.update_state(gp)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        features, real_labels = data
        real_labels = tf.cast(real_labels, tf.float32)
        gen_ctrs = self.generator(features, training=False)
        loss_ctr = self.loss_fn(real_labels, gen_ctrs)
        self.ctr_loss_tracker.update_state(loss_ctr)
        self.auc_tracker.update_state(real_labels, gen_ctrs)
        return {"val_ctr_loss": self.ctr_loss_tracker.result(), "val_g_auc": self.auc_tracker.result()}


# --- 6. 训练与评估 ---
# 加载数据
train_dataset = get_dataset(train_files, shuffle=True, num_epochs=EPOCHS, batch_size=BATCH_SIZE)
valid_dataset = get_dataset(valid_files, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)
test_dataset = get_dataset(test_files, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)

# 初始化模型、优化器和损失函数
generator = Generator(raw_features_dict)
discriminator = DiscriminatorWithMoE(  # <-- 使用新的 MoE 判别器
    raw_features_dict,
    num_experts=NUM_EXPERTS,
    expert_hidden_units=EXPERT_HIDDEN_UNITS
)

optimizer_g = keras.optimizers.Adam(learning_rate=LR_G)
optimizer_c = keras.optimizers.Adam(learning_rate=LR_D)
bce_loss = keras.losses.BinaryCrossentropy()

gan = GAN(generator=generator, discriminator=discriminator)
gan.compile(optimizer_g=optimizer_g, optimizer_d=optimizer_c, loss_fn=bce_loss)

print("\nStarting WGAN-GP training...")
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_ctr_loss', patience=1, verbose=1, mode='min', restore_best_weights=False)
]
history = gan.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset, callbacks=callbacks)

# --- 7. 保存与最终测试 ---
best_generator = gan.generator

# 保存最终的生成器模型用于线上服务
current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
final_model_path = os.path.join(serving_model_path, current_time)

# 为模型定义serving签名
@tf.function
def serving_function(data):
    return {'prob': best_generator(data)}


# 获取具体函数签名
input_spec = {}
for key, (max_vocab, dtype, emb_dim, f_type, is_used, desc) in raw_features_dict.items():
    if is_used:
        input_spec[key] = tf.TensorSpec(shape=[None, ], dtype=dtype, name=key)

concrete_serving_function = serving_function.get_concrete_function(input_spec)

print(f"Saving Generator model to {final_model_path}")
best_generator.save(final_model_path, overwrite=True, include_optimizer=False,
                    signatures={'serving_default': concrete_serving_function})

file = open(os.path.join(final_model_path, "_SUCCESS"), "w")
file.write("This is an example file created using Python!")
file.close()

print("\n--- Testing on test.csv using the trained Generator ---")

predictions = best_generator.predict(test_dataset)


def bylineread(filename):
    with open(filename, 'r') as f:
        # 跳过表头
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


final_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('Final time: ', final_time)