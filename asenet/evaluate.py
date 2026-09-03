"""评估指标计算与打印，复刻原脚本的 AUC / Logloss / PCOC 输出。"""
from sklearn.metrics import roc_auc_score, log_loss


def print_metrics(predictions, labels):
    """打印 test_losloss / test_auc / test_pcoc，逻辑与原脚本一致。"""
    if len(labels) > 0 and len(predictions) > 0:
        test_logloss = log_loss(labels, predictions)
        test_auc = roc_auc_score(labels, predictions)
        test_pcoc = sum(predictions)[0] / sum(labels) - 1 if sum(labels) > 0 else 0
        print("test_losloss:{:.6f}, test_auc:{:.6f}, test_pcoc:{:.6f}".format(
            test_logloss, test_auc, test_pcoc))
    else:
        print("Test data is empty, skipping evaluation.")
