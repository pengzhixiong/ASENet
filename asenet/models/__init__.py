"""模型注册表：将模型名映射到对应的构建函数。"""
from . import (asenet, deepfm, dien, din, dlf, dmr, final, gan,
               hierdiffuse, mirrn, qnn, twin, dnn)

# 标准判别模型：build_model(features_dict, ctx) -> keras.Model（未编译）
SUPERVISED_BUILDERS = {
    "dnn": dnn.build_model,
    "deepfm": deepfm.build_model,
    "din": din.build_model,
    "dien": dien.build_model,
    "dmr": dmr.build_model,
    "final": final.build_model,
    "twin": twin.build_model,
    "mirrn": mirrn.build_model,
    "qnn": qnn.build_model,
    "dlf": dlf.build_model,
    "hierdiffuse": hierdiffuse.build_model,
}

# GAN 体系模型：build_gan(features_dict, ctx) -> GAN 封装（未编译）
GAN_BUILDERS = {
    "gan": gan.build_gan,
    "asenet": asenet.build_gan,
}
