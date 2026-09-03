"""特征表、数据集与模型的元信息配置。

集中管理原来散布在 26 个脚本里的三样东西：
1. 特征定义（`raw_features_dict`）—— 两个数据集各有独立词表，其中 Taobao 存在
   「大词表 / 小词表」两套变体（影响 Hashing 桶数 → Embedding 表形状，必须严格保留）；
2. 数据集配置（label 列、文件路径、序列长度、序列/目标特征名）；
3. 模型元信息（是否启用序列特征、Taobao 词表变体、shuffle 覆盖等）。

特征仍用 6 元组表示 `(vocab, dtype, dim, is_seq, used, desc)`，与原始 `raw_features_dict`
的字段顺序完全一致，保证模型内 `spec[0..4]` 的索引语义不变。
"""
from collections import OrderedDict, namedtuple
from dataclasses import dataclass, field

import tensorflow as tf

# (vocab, dtype, dim, is_seq, used, desc)
Feature = namedtuple("Feature", "vocab dtype dim is_seq used desc")

# --------------------------------------------------------------------------- #
# 特征表（顺序与原始 raw_features_dict 严格一致）
# --------------------------------------------------------------------------- #
# Industrial 数据集：45 个特征。词表在所有模型间一致，仅序列特征 used 标志随模型变化。
INDUSTRIAL_FEATURES = [
    ("imp_rank", 10, tf.string, 8, 0, False, "iorder 第几次曝光"),
    ("buyer_sex", 10, tf.string, 8, 0, True, "buyer_sex 性别"),
    ("visit_chnl1", 50, tf.string, 8, 0, True, "visit_chnl1 注册一级渠道"),
    ("visit_chnl2", 500, tf.string, 8, 0, True, "visit_chnl2 注册二级渠道"),
    ("country_id", 500, tf.string, 8, 0, True, "countryId 国家"),
    ("site", 10, tf.string, 8, 0, True, "site 三端类型"),
    ("lang", 50, tf.string, 8, 0, True, "lang 语言"),
    ("hour", 72, tf.string, 8, 0, True, "hour 小时"),
    ("day_of_week", 35, tf.string, 8, 0, True, "weekDay 周几"),
    ("is_weekend", 5, tf.string, 8, 0, True, "isWeekend 是否周末"),
    ("searchword", 1000000, tf.string, 8, 0, True, "SearchWord 搜索词"),
    ("categorypred", 30000, tf.string, 8, 0, True, "CategoryPredict 类目预测(知识图谱)接口返回的类目"),
    ("visit_days_num_30d", 100, tf.string, 8, 0, True, "active_days_30d 活跃天数（最近30天)"),
    ("visit_prod_num_30d", 100, tf.string, 8, 0, True, "visit_item_num_30d 用户最近30天浏览商品数"),
    ("addfav_prod_num_30d", 100, tf.string, 8, 0, True, "favorite_item_num_30d 用户最近30天收藏商品数"),
    ("addcart_prod_num_30d", 100, tf.string, 8, 0, True, "cart_item_num_30d 用户最近30天加购商品数"),
    ("last_confirm_interval_etl_days", 100, tf.string, 8, 0, True, "recency 最后一次购买距离现在的天数"),
    ("confirm_days_frequency_his", 100, tf.string, 8, 0, True, "frequency 购买频率(周期)"),
    ("started_rfx_num_365d", 100, tf.string, 8, 0, True, "total_order_cnt 用户累计订单量"),
    ("started_rfx_num_30d", 100, tf.string, 8, 0, True, "started_item_num_30d 用户最近30天下单商品数"),
    ("visit_cate1_pub_max_30d", 300, tf.string, 8, 0, True, "Visit_cate1_30d 用户最近30天浏览主要一级类目"),
    ("addcart_cate1_pub_max_30d", 300, tf.string, 8, 0, True, "Cart_cate1_30d 用户最近30天加购主要一级类目"),
    ("confirm_cate1_pub_num_30d", 300, tf.string, 8, 0, True, "Confirmorder_cate1_30d 用户最近30天付款确认主要一级类目"),
    ("itemcode", 1000000, tf.string, 8, 0, True, "item_code 商品id"),
    ("seller_id", 20000, tf.string, 8, 0, True, "seller_id 卖家id"),
    ("cate_disp_id", 5000, tf.string, 8, 0, True, "cate_disp_id 展示类目id"),
    ("is_free_ship", 5, tf.string, 8, 0, True, "is_free_ship 是否包邮"),
    ("is_newuseronly", 5, tf.string, 8, 0, True, "is_newuseronly 是否是正在参加新人专享价商品(1新人折扣)"),
    ("expo_1d", 100, tf.string, 8, 0, True, "expo_1d 商品1天内曝光"),
    ("expo_7d", 200, tf.string, 8, 0, True, "expo_7d 商品7天内曝光"),
    ("expo_30d", 300, tf.string, 8, 0, True, "expo_30d 商品30天曝光数"),
    ("click_1d", 100, tf.string, 8, 0, True, "click_1d 商品1天内点击"),
    ("click_7d", 100, tf.string, 8, 0, True, "click_7d 商品7天内点击"),
    ("click_30d", 100, tf.string, 8, 0, True, "click_30d 商品30天内点击"),
    ("favorite_30d", 100, tf.string, 8, 0, True, "favorite_30d 商品收藏次数_30d "),
    ("addcart_30d", 100, tf.string, 8, 0, True, "addcart_30d 商品加购次数_30d "),
    ("order_30d", 100, tf.string, 8, 0, True, "order_30d 商品累计出单总数30d "),
    ("order_all", 100, tf.string, 8, 0, True, "order_all 商品总成单数 "),
    ("review_all", 100, tf.string, 8, 0, True, "review_all 商品历史总评论数 "),
    ("search_expo_list_30d", 800000, tf.string, 8, 1, False, "30天曝光的广告ID序列"),
    ("search_click_list_30d", 800000, tf.string, 8, 1, False, "30天点击的商品ID序列"),
    ("search_cate_disp_list_30d", 800000, tf.string, 8, 1, True, "30天曝光的广告ID序列"),
    ("search_seller_list_30d", 800000, tf.string, 8, 1, True, "30天点击的商品ID序列"),
    ("search_query_list_30d", 800000, tf.string, 8, 1, False, "30天点击的商品ID序列"),
    ("is_click", 100, tf.float32, 8, 0, False, "is_click 是否点击"),
]

# Taobao 数据集：21 个特征。默认采用「大词表」，另有 6 个特征在小词表变体中缩小。
TAOBAO_FEATURES_LARGE = [
    ("clk", 10, tf.float32, 8, 0, False, "是否点击，训练的label"),
    ("btag_his", 10, tf.string, 8, 1, False, "行为类型序列，包括ipv/cart/fav/buy"),
    ("cate_his", 15000, tf.string, 8, 1, True, "历史点击广告商品的类别ID序列"),
    ("brand_his", 200000, tf.string, 8, 1, True, "历史点击广告商品的品牌ID序列"),
    ("userid", 1500000, tf.string, 8, 0, True, "用户ID"),
    ("cms_segid", 300, tf.string, 8, 0, True, "微群ID"),
    ("cms_group_id", 50, tf.string, 8, 0, True, "微群组ID"),
    ("final_gender_code", 10, tf.string, 8, 0, True, "性别"),
    ("age_level", 35, tf.string, 8, 0, True, "年龄层次"),
    ("pvalue_level", 20, tf.string, 8, 0, True, "消费档次，低档/中档/高档"),
    ("shopping_level", 15, tf.string, 8, 0, True, "购物深度，浅层/中度/深度"),
    ("occupation", 10, tf.string, 8, 0, True, "职业，是否大学生1：是，0：否"),
    ("new_user_class_level", 25, tf.string, 8, 0, True, "城市级别"),
    ("adgroup_id", 1200000, tf.string, 8, 0, True, "广告ID"),
    ("cate_id", 15000, tf.string, 8, 0, True, "当前广告商品的类别ID"),
    ("campaign_id", 800000, tf.string, 8, 0, True, "广告计划ID"),
    ("customer", 500000, tf.string, 8, 0, True, "广告主ID"),
    ("brand", 200000, tf.string, 8, 0, True, "当前广告商品的品牌ID"),
    ("price", 300, tf.float32, 8, 0, True, "商品价格，已归一化 0～1"),
    ("pid", 10, tf.string, 8, 0, True, "资源位id"),
    ("btag", 10, tf.string, 8, 0, False, "行为类型，包括ipv/cart/fav/buy"),
]

# 小词表相对大词表的 vocab 覆盖（原实验中 DIN/DLF/DNN/GAN/HierDiffuse/QNN 使用小词表）。
TAOBAO_SMALL_VOCAB = {
    "userid": 1000000,
    "adgroup_id": 800000,
    "campaign_id": 400000,
    "customer": 300000,
    "brand": 150000,
    "brand_his": 150000,
}

# 每个数据集会被静态模型（DNN/DeepFM/GAN）关闭的序列特征。
_SEQ_FEATURES = {
    "industrial": ["search_cate_disp_list_30d", "search_seller_list_30d"],
    "taobao": ["cate_his", "brand_his"],
}


def build_feature_dict(dataset, use_seq=True, taobao_vocab="large"):
    """构建与原始 raw_features_dict 顺序一致的 OrderedDict。

    :param dataset: 'industrial' | 'taobao'
    :param use_seq: 是否启用序列特征（静态模型传 False）
    :param taobao_vocab: 'large' | 'small'（仅对 taobao 生效）
    """
    rows = INDUSTRIAL_FEATURES if dataset == "industrial" else TAOBAO_FEATURES_LARGE
    feature_dict = OrderedDict()
    for name, vocab, dtype, dim, is_seq, used, desc in rows:
        if dataset == "taobao" and taobao_vocab == "small" and name in TAOBAO_SMALL_VOCAB:
            vocab = TAOBAO_SMALL_VOCAB[name]
        if not use_seq and name in _SEQ_FEATURES[dataset]:
            used = False
        feature_dict[name] = Feature(vocab, dtype, dim, is_seq, used, desc)
    return feature_dict


# --------------------------------------------------------------------------- #
# 数据集配置
# --------------------------------------------------------------------------- #
DATASETS = {
    "industrial": {
        "label_col": "is_click",
        "files": {
            "train": "/data2/jupyter/nobackup/dataset/searchv6/train_20250924_ns.csv",
            "valid": "/data2/jupyter/nobackup/dataset/searchv6/valid_20250924_ns.csv",
            "test": "/data2/jupyter/nobackup/dataset/searchv6/fmsearch_20250924_shuf.csv",
            "serving": "/data2/jupyter/nobackup/model/search_ctr_gan_v1/",
        },
        "max_seq_len": 20,
        "short_seq_len": 5,
        "local_seq_len": 5,
        # 序列 / 目标特征名（第二个序列特征的语义为 seller）
        "seq_cate": "search_cate_disp_list_30d",
        "seq_brand": "search_seller_list_30d",
        "target_cate": "cate_disp_id",
        "target_brand": "seller_id",
        "default_shuffle": 100000,
    },
    "taobao": {
        "label_col": "clk",
        "files": {
            "train": "/data2/jupyter/nobackup/dataset/searchv6/train_tb.csv",
            "valid": "/data2/jupyter/nobackup/dataset/searchv6/valid_tb.csv",
            "test": "/data2/jupyter/nobackup/dataset/searchv6/test.csv",
            "serving": "/data2/jupyter/nobackup/model/search_ctr_gan_v1/",
        },
        "max_seq_len": 50,
        "short_seq_len": 10,
        "local_seq_len": 10,
        "seq_cate": "cate_his",
        "seq_brand": "brand_his",
        "target_cate": "cate_id",
        "target_brand": "brand",
        "default_shuffle": 100000,
    },
}

# ASENet 的共享嵌入分组（组名 / base 特征 / 成员特征），每个数据集各一套，严格保留原映射。
ASENET_SHARE_GROUPS = {
    "industrial": {
        "cate": {"base": "cate_disp_id", "members": ["cate_disp_id", "search_cate_disp_list_30d"]},
        "seller": {"base": "seller_id", "members": ["seller_id", "search_seller_list_30d"]},
    },
    "taobao": {
        "cate": {"base": "cate_id", "members": ["cate_id", "cate_his"]},
        "brand": {"base": "brand", "members": ["brand", "brand_his"]},
    },
}

# --------------------------------------------------------------------------- #
# 模型元信息
# --------------------------------------------------------------------------- #
# use_seq: 是否启用序列特征；taobao_vocab: Taobao 词表变体；
# industrial_shuffle: 覆盖 Industrial 数据集的 shuffle 大小（缺省用 default_shuffle）。
MODELS = {
    "dnn":         {"use_seq": False, "taobao_vocab": "small"},
    "deepfm":      {"use_seq": False, "taobao_vocab": "large"},
    "din":         {"use_seq": True,  "taobao_vocab": "small"},
    "dien":        {"use_seq": True,  "taobao_vocab": "large"},
    "dmr":         {"use_seq": True,  "taobao_vocab": "large"},
    "final":       {"use_seq": True,  "taobao_vocab": "large", "industrial_shuffle": 10000},
    "twin":        {"use_seq": True,  "taobao_vocab": "large", "industrial_shuffle": 10000},
    "mirrn":       {"use_seq": True,  "taobao_vocab": "large", "industrial_shuffle": 10000},
    "gan":         {"use_seq": False, "taobao_vocab": "small"},
    "qnn":         {"use_seq": True,  "taobao_vocab": "small"},
    "dlf":         {"use_seq": True,  "taobao_vocab": "small"},
    "hierdiffuse": {"use_seq": True,  "taobao_vocab": "small"},
    "asenet":      {"use_seq": True,  "taobao_vocab": "large", "industrial_shuffle": 10000},
}

# 属于 GAN 体系的模型（走 GAN 训练 + 生成器导出路径）。
GAN_MODELS = {"gan", "asenet"}


@dataclass
class Context:
    """一次实验所需的全部配置，由 build_context 汇总生成。"""
    dataset: str
    model: str
    feature_dict: "OrderedDict"          # 完整特征表（含 used 标志）
    label_col: str
    files: dict
    shuffle_size: int
    max_seq_len: int
    short_seq_len: int
    local_seq_len: int
    seq_cate: str
    seq_brand: str
    target_cate: str
    target_brand: str
    # 通用超参
    batch_size: int = 2048
    epochs: int = 1
    learning_rate: float = 0.001

    @property
    def used_feature_dict(self):
        """仅包含 used=True 的特征（等价于原始 features_dict）。"""
        return OrderedDict((k, v) for k, v in self.feature_dict.items() if v.used)

    @property
    def label_idx(self):
        return 0 if self.label_col == "clk" else -1


def build_context(dataset, model):
    """根据数据集与模型名组装 Context。"""
    if dataset not in DATASETS:
        raise ValueError(f"未知数据集: {dataset}")
    if model not in MODELS:
        raise ValueError(f"未知模型: {model}")

    ds = DATASETS[dataset]
    meta = MODELS[model]

    feature_dict = build_feature_dict(
        dataset, use_seq=meta["use_seq"], taobao_vocab=meta.get("taobao_vocab", "large")
    )

    shuffle_size = ds["default_shuffle"]
    if dataset == "industrial" and "industrial_shuffle" in meta:
        shuffle_size = meta["industrial_shuffle"]

    return Context(
        dataset=dataset,
        model=model,
        feature_dict=feature_dict,
        label_col=ds["label_col"],
        files=ds["files"],
        shuffle_size=shuffle_size,
        max_seq_len=ds["max_seq_len"],
        short_seq_len=ds["short_seq_len"],
        local_seq_len=ds["local_seq_len"],
        seq_cate=ds["seq_cate"],
        seq_brand=ds["seq_brand"],
        target_cate=ds["target_cate"],
        target_brand=ds["target_brand"],
    )
