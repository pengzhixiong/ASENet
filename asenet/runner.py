"""实验编排：构建配置/数据/模型，训练、评估并导出线上模型。"""
import tensorflow as tf
import tensorflow.keras as keras

from .config import GAN_MODELS, build_context
from .data import build_column_specs, get_dataset, read_labels
from .evaluate import print_metrics
from .models import GAN_BUILDERS, SUPERVISED_BUILDERS
from .serve import build_signature, save_model

# 保存后 reload 再预测的模型（复刻原脚本的 load_model 评估路径）
RELOAD_FOR_EVAL = {"final", "twin", "mirrn", "qnn", "dlf", "hierdiffuse"}
# serving 签名以 `model(data, training=False)` 调用的模型
USE_TRAINING_FLAG = {"final", "qnn"}


def _build_datasets(ctx, csv_columns, select_columns, column_defaults):
    """构建 train / valid / test 三个数据集。"""
    train_dataset = get_dataset(
        ctx.files["train"], csv_columns, select_columns, column_defaults,
        ctx.label_col, ctx.shuffle_size, shuffle=True, num_epochs=ctx.epochs, batch_size=ctx.batch_size)
    valid_dataset = get_dataset(
        ctx.files["valid"], csv_columns, select_columns, column_defaults,
        ctx.label_col, ctx.shuffle_size, shuffle=False, num_epochs=1, batch_size=ctx.batch_size)
    test_dataset = get_dataset(
        ctx.files["test"], csv_columns, select_columns, column_defaults,
        ctx.label_col, ctx.shuffle_size, shuffle=False, num_epochs=1, batch_size=ctx.batch_size)
    return train_dataset, valid_dataset, test_dataset


def _run_supervised(ctx, train_dataset, valid_dataset, test_dataset):
    """标准判别模型：compile -> fit -> save -> 评估。"""
    model = SUPERVISED_BUILDERS[ctx.model](ctx.used_feature_dict, ctx)

    optimizer = tf.keras.optimizers.Adam(learning_rate=ctx.learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    auc = tf.keras.metrics.AUC(name="my_auc")

    signature = build_signature(model, ctx.used_feature_dict,
                                use_training_flag=(ctx.model in USE_TRAINING_FLAG))

    model.compile(optimizer=optimizer, loss=loss, metrics=[auc])
    history = model.fit(train_dataset, epochs=ctx.epochs, verbose=1, validation_data=valid_dataset)

    saved_path = save_model(model, ctx.files["serving"], signature)

    if ctx.model in RELOAD_FOR_EVAL:
        new_model = tf.keras.models.load_model(saved_path)
        predictions = new_model.predict(test_dataset)
    else:
        predictions = model.predict(test_dataset)

    labels = read_labels(ctx.files["test"], ctx.label_idx)
    print_metrics(predictions, labels)


def _run_gan(ctx, train_dataset, valid_dataset, test_dataset):
    """GAN 体系模型：GAN 封装训练，导出生成器用于线上服务。"""
    gan = GAN_BUILDERS[ctx.model](ctx.feature_dict, ctx)

    optimizer_g = keras.optimizers.Adam(learning_rate=ctx.learning_rate)
    optimizer_d = keras.optimizers.Adam(learning_rate=ctx.learning_rate)
    bce_loss = keras.losses.BinaryCrossentropy()

    gan.compile(optimizer_g=optimizer_g, optimizer_d=optimizer_d, loss_fn=bce_loss)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_ctr_loss", patience=1, verbose=1,
                                      mode="min", restore_best_weights=False)
    ]
    gan.fit(train_dataset, epochs=ctx.epochs, validation_data=valid_dataset, callbacks=callbacks)

    generator = gan.generator

    signature = build_signature(generator, ctx.used_feature_dict, use_training_flag=False)
    # 仅 Industrial ASENet 额外写 _SUCCESS 标记文件（复刻原始脚本差异）
    save_model(generator, ctx.files["serving"], signature,
               write_success=(ctx.model == "asenet" and ctx.dataset == "industrial"))

    predictions = generator.predict(test_dataset)
    labels = read_labels(ctx.files["test"], ctx.label_idx)
    print_metrics(predictions, labels)


def run_model(dataset, model):
    """运行单个 (数据集, 模型) 实验。"""
    ctx = build_context(dataset, model)

    csv_columns, select_columns, column_defaults = build_column_specs(ctx.feature_dict, ctx.label_col)
    print("_CSV_COLUMNS: {}, _SELECT_COLUMNS: {}".format(len(csv_columns), len(select_columns)))

    train_dataset, valid_dataset, test_dataset = _build_datasets(
        ctx, csv_columns, select_columns, column_defaults)

    if model in GAN_MODELS:
        print(f"\nStarting {model.upper()} (GAN) training on {dataset}...")
        _run_gan(ctx, train_dataset, valid_dataset, test_dataset)
    else:
        print(f"\nStart Training {model.upper()} on {dataset}...")
        _run_supervised(ctx, train_dataset, valid_dataset, test_dataset)
