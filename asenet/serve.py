"""线上导出：构建 serving 签名并保存模型。"""
import os

import tensorflow as tf


def build_signature(model, feature_dict, use_training_flag=False):
    """构建 serving 签名，返回 concrete function。

    :param use_training_flag: 是否以 `model(data, training=False)` 调用（FINAL / QNN 需要）。
    """
    @tf.function
    def serving_function(data):
        if use_training_flag:
            return {"prob": model(data, training=False)}
        return {"prob": model(data)}

    input_spec = {}
    for key, spec in feature_dict.items():
        input_spec[key] = tf.TensorSpec(shape=[None, ], dtype=spec.dtype, name=key)
    return serving_function.get_concrete_function(input_spec)


def save_model(model, serving_dir, concrete_function, write_success=False):
    """以 `serving_default` 签名保存模型（`include_optimizer=False`）。

    返回实际保存路径（含时间戳子目录）。
    """
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    final_model_path = os.path.join(serving_dir, current_time)
    print(f"Saving model to {final_model_path}")
    model.save(
        final_model_path,
        overwrite=True,
        include_optimizer=False,
        signatures={"serving_default": concrete_function},
    )
    if write_success:
        with open(os.path.join(final_model_path, "_SUCCESS"), "w") as f:
            f.write("This is an example file created using Python!")
    return final_model_path
