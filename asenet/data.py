"""CSV 数据加载与标签读取，复刻原脚本的 get_dataset / bylineread 逻辑。"""
import tensorflow as tf


def build_column_specs(feature_dict, label_col):
    """根据特征表构建 make_csv_dataset 所需的列名 / 选择列 / 默认值。

    等价于原脚本中构建 _CSV_COLUMNS / _SELECT_COLUMNS / _COLUMN_DEFAULTS 的循环。
    """
    csv_columns = []
    select_columns = []
    column_defaults = []
    for key, spec in feature_dict.items():
        csv_columns.append(key)
        if spec.used or key == label_col:
            select_columns.append(key)
            if spec.dtype is tf.int32:
                column_defaults.append(tf.constant(0, dtype=spec.dtype))
            elif spec.dtype is tf.float32:
                column_defaults.append(tf.constant(0.0, dtype=spec.dtype))
            else:
                column_defaults.append(tf.constant("", dtype=spec.dtype))
    return csv_columns, select_columns, column_defaults


def get_dataset(data_path, csv_columns, select_columns, column_defaults, label_col,
                shuffle_size, shuffle=True, num_epochs=1, batch_size=512):
    """构建 tf.data 数据集，参数与原 get_dataset 逐项一致。"""
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=data_path,
        batch_size=batch_size,
        column_names=csv_columns,
        column_defaults=column_defaults,
        label_name=label_col,
        select_columns=select_columns,
        field_delim=",",
        header=True,
        num_epochs=num_epochs,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_size,
        prefetch_buffer_size=tf.data.AUTOTUNE,
        num_parallel_reads=tf.data.AUTOTUNE,
        shuffle_seed=42,
    )
    return dataset


def read_labels(test_files, label_idx):
    """逐行读取测试集真实标签（跳过表头），复刻原 bylineread 逻辑。"""

    def bylineread(filename):
        with open(filename, "r") as f:
            next(f)  # 跳过表头
            for line in f:
                yield line

    return [int(line.strip().split(",")[label_idx]) for line in bylineread(test_files)]
