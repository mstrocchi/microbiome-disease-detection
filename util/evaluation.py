import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, RocCurveDisplay, PrecisionRecallDisplay, roc_auc_score


def auc_metrics(y, y_hat):
    precision, recall, thresholds = precision_recall_curve(y, y_hat)
    return {
        'ROC AUC': roc_auc_score(y, y_hat),
        'PR AUC': auc(recall, precision)
    }


def plot_loss_curve(losses):
    plt.plot(losses, label="Set Transformer")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(axis='y')
    plt.show()


def plot_roc(y, y_hat, instance_name):
    RocCurveDisplay.from_predictions(y, y_hat, name=instance_name)
    plt.plot([0, 1], [0, 1], "k--")
    plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(np.arange(0, 1.1, .1))
    plt.grid(axis='both')
    plt.show()


def plot_pr(y, y_hat, instance_name):
    PrecisionRecallDisplay.from_predictions(y, y_hat, name=instance_name)
    plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(np.arange(0, 1.1, .1))
    plt.grid(axis='both')
    plt.show()


def evaluation(y, y_hat, instance_name):
    y = y.flatten()
    y_hat = y_hat.detach().numpy()
    plot_roc(y, y_hat, instance_name)
    plot_pr(y, y_hat, instance_name)
    print(auc_metrics(y, y_hat))
