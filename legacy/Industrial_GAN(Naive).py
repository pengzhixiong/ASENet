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
raw_features_dict['search_click_list_30d'] = (800000, tf.string, 8, 1, False, '30天点击的商品ID序列') # Changed type to 1 for multi-value
raw_features_dict['search_cate_disp_list_30d'] = (800000, tf.string, 8, 1, False, '30天曝光的广告ID序列')
raw_features_dict['search_seller_list_30d'] = (800000, tf.string, 8, 1, False, '30天点击的商品ID序列')
raw_features_dict['search_query_list_30d'] = (800000, tf.string, 8, 1, False, '30天点击的商品ID序列')
raw_features_dict['is_click'] = (100, tf.float32, 8, 0, False, 'is_click 是否点击 ')

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
LR_G = 0.001
LR_D = 0.001
CONTENT_LOSS_WEIGHT = 20.0


class FeatureExtractor(keras.Model):
    """特征提取器基类，封装共享的特征处理逻辑。"""
    def __init__(self, features_dict, **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs)
        self.features_dict = {k: v for k, v in features_dict.items() if v[4]}
        self.hash_layers = {}
        self.embedding_layers = {}
        for name, (max_vocab, dtype, emb_dim, f_type, is_used, _) in self.features_dict.items():
            if dtype == tf.string:
                self.hash_layers[name] = keras.layers.Hashing(num_bins=max_vocab, name=f'{name}_hash')
                self.embedding_layers[name] = keras.layers.Embedding(
                    input_dim=max_vocab, output_dim=emb_dim, name=f'{name}_embedding'
                )

    def get_embedding_list(self, inputs):
        """处理输入特征，返回一个包含所有特征嵌入向量的列表。"""
        processed_features = []
        for name, (_, dtype, _, f_type, _, _) in self.features_dict.items():
            feature_tensor = inputs[name]
            if dtype == tf.string:
                if f_type == 1:
                    split_feature = tf.strings.split(feature_tensor, sep='^')
                    hashed_feature = self.hash_layers[name](split_feature)
                    embedded_feature = self.embedding_layers[name](hashed_feature)
                    pooled_feature = tf.reduce_mean(embedded_feature, axis=1)
                    processed_features.append(pooled_feature)
                else:
                    hashed_feature = self.hash_layers[name](feature_tensor)
                    embedded_feature = self.embedding_layers[name](hashed_feature)
                    processed_features.append(embedded_feature)
        return processed_features

    def call(self, inputs):
        """处理输入特征并拼接成一个扁平的向量。"""
        embedding_list = self.get_embedding_list(inputs)
        return keras.layers.concatenate(embedding_list, axis=-1)

class Generator(FeatureExtractor):
    """生成器：标准的CTR预估模型"""
    def __init__(self, features_dict, **kwargs):
        super(Generator, self).__init__(features_dict, **kwargs)
        self.mlp = tf.keras.Sequential([
            keras.layers.Dense(128, activation='relu', name='g_dense_1'),
            keras.layers.Dense(64, activation='relu', name='g_dense_2'),
            keras.layers.Dense(1, activation='sigmoid', name='g_output')
        ], name='GeneratorMLP')

    def call(self, inputs):
        feature_vector = super(Generator, self).call(inputs)
        return self.mlp(feature_vector)

class Discriminator(FeatureExtractor):
    """判别器：判断 (特征, CTR) 对的真实性"""
    def __init__(self, features_dict, **kwargs):
        super(Discriminator, self).__init__(features_dict, **kwargs)
        self.mlp = tf.keras.Sequential([
            keras.layers.Dense(128, activation='relu', name='d_dense_1'),
            keras.layers.Dense(64, activation='relu', name='d_dense_2'),
            keras.layers.Dense(1, activation=None, name='d_output')
        ], name='DiscriminatorMLP')

    def call(self, inputs):
        features, ctr = inputs
        feature_vector = super(Discriminator, self).call(features)
        ctr_reshaped = tf.reshape(ctr, shape=(-1, 1))
        
        combined_input = keras.layers.concatenate([feature_vector, ctr_reshaped], axis=-1)
        return self.mlp(combined_input)

class GAN(keras.Model):
    """GAN模型，封装了G和D以及自定义的训练逻辑"""
    def __init__(self, generator, discriminator, **kwargs):
        super(GAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.ctr_loss_tracker = keras.metrics.Mean(name="ctr_loss")
        self.adv_loss_tracker = keras.metrics.Mean(name="adv_loss")
        self.auc_tracker = tf.keras.metrics.AUC(name='g_auc')

    def compile(self, optimizer_g, optimizer_d, loss_fn):
        super(GAN, self).compile()
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker, self.ctr_loss_tracker, self.adv_loss_tracker, self.auc_tracker]

    @tf.function
    def train_step(self, data):
        features, real_labels = data
        real_labels = tf.cast(real_labels, tf.float32)
        batch_size = tf.shape(real_labels)[0]
        
        valid = tf.ones((batch_size, 1)) * 0.9
        fake = tf.ones((batch_size, 1)) * 0.1


        with tf.GradientTape() as tape:
            real_pred = self.discriminator([features, real_labels])
            d_loss_real = self.loss_fn(valid, real_pred)
            
            gen_ctrs = self.generator(features, training=True)
            fake_pred = self.discriminator([features, gen_ctrs])
            d_loss_fake = self.loss_fn(fake, fake_pred)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
        
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.optimizer_d.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))


        with tf.GradientTape() as tape:
            gen_ctrs = self.generator(features, training=True)
            validity = self.discriminator([features, gen_ctrs], training=True)
            
            loss_ctr = self.loss_fn(real_labels, gen_ctrs)
            
            loss_adv = self.loss_fn(valid, validity)
            

            g_loss = CONTENT_LOSS_WEIGHT * loss_ctr + loss_adv


        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.optimizer_g.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.ctr_loss_tracker.update_state(loss_ctr)
        self.adv_loss_tracker.update_state(loss_adv)
        self.auc_tracker.update_state(real_labels, gen_ctrs)
        
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        features, real_labels = data
        real_labels = tf.cast(real_labels, tf.float32)
        gen_ctrs = self.generator(features, training=False)
        loss_ctr = self.loss_fn(real_labels, gen_ctrs)
        self.ctr_loss_tracker.update_state(loss_ctr)
        self.auc_tracker.update_state(real_labels, gen_ctrs)
        return {"val_ctr_loss": self.ctr_loss_tracker.result(), "val_g_auc": self.auc_tracker.result()}



train_dataset = get_dataset(train_files, shuffle=True, num_epochs=EPOCHS, batch_size=BATCH_SIZE)
valid_dataset = get_dataset(valid_files, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)
test_dataset = get_dataset(test_files, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)


generator = Generator(raw_features_dict)
discriminator = Discriminator(raw_features_dict)

optimizer_g = keras.optimizers.Adam(learning_rate=LR_G)
optimizer_d = keras.optimizers.Adam(learning_rate=LR_D)
bce_loss = keras.losses.BinaryCrossentropy()

gan = GAN(generator=generator, discriminator=discriminator)
gan.compile(optimizer_g=optimizer_g, optimizer_d=optimizer_d, loss_fn=bce_loss)

print("\nStarting GAN training...")
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_ctr_loss', patience=1, verbose=1, mode='min', restore_best_weights=False)
]
history = gan.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset, callbacks=callbacks)


best_generator = gan.generator
print("\n--- Testing on test.csv using the trained Generator ---")

predictions = best_generator.predict(test_dataset)

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


current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
final_model_path = os.path.join(serving_model_path, current_time)


@tf.function
def serving_function(data):
    return {'prob': best_generator(data)}


input_spec = {}
for key, (max_vocab, dtype, emb_dim, f_type, is_used, desc) in raw_features_dict.items():
    if is_used:
        input_spec[key] = tf.TensorSpec(shape=[None,], dtype=dtype, name=key)
        
concrete_serving_function = serving_function.get_concrete_function(input_spec)

print(f"Saving Generator model to {final_model_path}")
best_generator.save(final_model_path, overwrite=True, include_optimizer=False,
                    signatures={'serving_default': concrete_serving_function})

final_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('Final time: ', final_time)
