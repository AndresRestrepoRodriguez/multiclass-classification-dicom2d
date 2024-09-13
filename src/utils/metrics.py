from typing import Dict, List

import numpy as np
from scipy.special import rel_entr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def confusionmatrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Confusion Matrix for multiclass task.

    Args:
        y_true (np.ndarray): ground truth
        y_pred (np.ndarray): predicted probabilities/scores

    Returns:
        np.ndarray: confusion matrix as in sklearn
    """
    yp = np.argmax(y_pred, axis=1)
    return confusion_matrix(y_true, yp)


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes macro, micro and weighted Specificity.

    Args:
        y_true (np.ndarray): ground truth
        y_pred (np.ndarray): predicted probabilities/scores

    Returns:
        Dict[str, float]: a dictionary with keys ["Micro", "Macro", "Weighted"]
            containing the corresponding values
    """
    weights = np.unique(y_true, return_counts=True)[1]
    yp = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, yp)
    num_classes = cm.shape[0]
    tp_list = [cm[i, i] for i in range(num_classes)]
    fp_list = [np.sum(cm[:, i]) - cm[i, i] for i in range(num_classes)]
    fn_list = [np.sum(cm[i, :]) - cm[i, i] for i in range(num_classes)]
    tn_list = [
        np.sum(cm) - tp_list[i] - fp_list[i] - fn_list[i] for i in range(num_classes)
    ]

    tn_sum = np.sum(tn_list)
    fp_sum = np.sum(fp_list)

    micro_specificity = tn_sum / (tn_sum + fp_sum)
    macro_specificity = np.mean([(tn) / (tn + fp) for tn, fp in zip(tn_list, fp_list)])
    weighted_specificity = np.sum(
        [(tn) / (tn + fp) * w for tn, fp, w in zip(tn_list, fp_list, weights)]
    ) / np.sum(weights)
    return {
        "Micro": micro_specificity,
        "Macro": macro_specificity,
        "Weighted": weighted_specificity,
    }


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multiclass Classification Accuracy

    Args:
        y_true (np.ndarray): numpy array containing the ground truth
        y_pred (np.ndarray): numpy array containing the predictions

    Returns:
        float: the accuracy of the predictions
    """
    yp = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, yp)


def f1score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes macro, micro and weighted F1 score.

    Args:
        y_true (np.ndarray): ground truth
        y_pred (np.ndarray): predicted proba or scores

    Returns:
        Dict[str, float]: a dictionary with keys ["Micro", "Macro", "Weighted"]
            containing the corresponding values
    """
    yp = np.argmax(y_pred, axis=1)
    prec = {
        "Micro": f1_score(y_true, yp, average="micro"),
        "Macro": f1_score(y_true, yp, average="macro"),
        "Weighted": f1_score(y_true, yp, average="weighted"),
    }
    return prec


def f05score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes macro, micro and weighted F0.5 score.

    Args:
        y_true (np.ndarray): ground truth
        y_pred (np.ndarray): predicted proba or scores

    Returns:
        Dict[str, float]: a dictionary with keys ["Micro", "Macro", "Weighted"]
            containing the corresponding values
    """
    yp = np.argmax(y_pred, axis=1)
    prec = {
        "Micro": fbeta_score(y_true, yp, average="micro", beta=0.5),
        "Macro": fbeta_score(y_true, yp, average="macro", beta=0.5),
        "Weighted": fbeta_score(y_true, yp, average="weighted", beta=0.5),
    }
    return prec


def f2score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes macro, micro and weighted F2 score.

    Args:
        y_true (np.ndarray): ground truth
        y_pred (np.ndarray): predicted proba or scores

    Returns:
        Dict[str, float]: a dictionary with keys ["Micro", "Macro", "Weighted"]
            containing the corresponding values
    """
    yp = np.argmax(y_pred, axis=1)
    prec = {
        "Micro": fbeta_score(y_true, yp, average="micro", beta=2),
        "Macro": fbeta_score(y_true, yp, average="macro", beta=2),
        "Weighted": fbeta_score(y_true, yp, average="weighted", beta=2),
    }
    return prec


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes macro, micro and weighted Precision metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth
        y_pred (np.ndarray): numpy array containing the predictions

    Returns:
        Dict[str, float]: a dictionary containing micro, macro and
            weighted precision
    """
    yp = np.argmax(y_pred, axis=1)
    prec = {
        "Micro": precision_score(y_true, yp, average="micro"),
        "Macro": precision_score(y_true, yp, average="macro"),
        "Weighted": precision_score(y_true, yp, average="weighted"),
    }
    return prec


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes macro, micro and weighted Recall metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth
        y_pred (np.ndarray): numpy array containing the predictions

    Returns:
        Dict[str, float]: a dictionary containing micro, macro and
            weighted precision
    """
    yp = np.argmax(y_pred, axis=1)
    rec = {
        "Micro": recall_score(y_true, yp, average="micro"),
        "Macro": recall_score(y_true, yp, average="macro"),
        "Weighted": recall_score(y_true, yp, average="weighted"),
    }
    return rec


def kl_divergence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multiclass KL-Divergence.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth
        y_pred (np.ndarray): numpy array containing the predictions

    Returns:
        float: KL-Divergence
    """
    yp = y_pred >= 0.5
    p = np.array(np.unique(y_true, return_counts=True)[1], dtype=float)
    q = np.array(np.unique(yp, return_counts=True)[1], dtype=float)
    p /= np.array(np.sum(p), dtype=float)
    q /= np.array(np.sum(q), dtype=float)
    p, q = q, p
    return np.sum(rel_entr(p, q))


#######################################################################################
# Per class metrics
#######################################################################################


def per_class_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per class Specificity metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth
        y_pred (np.ndarray): numpy array containing the predictions

    Returns:
        List[float]: a list containing specificity per class
    """
    yp = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, yp)
    num_classes = cm.shape[0]
    specificity_score = []
    for class_index in range(num_classes):
        true_negative = (
            np.sum(cm)
            - np.sum(cm[class_index, :])
            - np.sum(cm[:, class_index])
            + cm[class_index, class_index]
        )
        false_positive = np.sum(cm[:, class_index]) - cm[class_index, class_index]
        specificity_score.append(
            round(true_negative / (true_negative + false_positive), 4)
        )
    return np.array(specificity_score)


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    """Per class Precision metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth
        y_pred (np.ndarray): numpy array containing the predictions

    Returns:
        List[float]: a list containing precision per class
    """
    yp = np.argmax(y_pred, axis=1)
    n_classes = y_pred.shape[1]
    acc = [
        accuracy_score(np.array(y_true) == i, np.array(yp) == i)
        for i in range(n_classes)
    ]
    return acc


def per_class_precision(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    """Per class Precision metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth
        y_pred (np.ndarray): numpy array containing the predictions

    Returns:
        List[float]: a list containing precision per class
    """
    yp = np.argmax(y_pred, axis=1)
    n_classes = y_pred.shape[1]

    pr_score = []
    for cl in range(n_classes):
        pr_score.append(precision_score(y_true, yp, average=None, labels=[cl])[0])

    return pr_score


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    """Per class Recall metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth
        y_pred (np.ndarray): numpy array containing the predictions

    Returns:
        List[float]: a list containing recall per class
    """
    yp = np.argmax(y_pred, axis=1)
    n_classes = y_pred.shape[1]

    rec_score = []
    for cl in range(n_classes):
        rec_score.append(recall_score(y_true, yp, average=None, labels=[cl])[0])

    return rec_score


def per_class_f1score(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    """Per class F1 Score metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth
        y_pred (np.ndarray): numpy array containing the predictions

    Returns:
        List[float]: a list containing f1-score per class
    """
    yp = np.argmax(y_pred, axis=1)
    n_classes = y_pred.shape[1]

    f1_scores = []
    for cl in range(n_classes):
        f1_scores.append(f1_score(y_true, yp, average=None, labels=[cl])[0])

    return f1_scores


def per_class_f2score(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    """Per class F1 Score metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth
        y_pred (np.ndarray): numpy array containing the predictions

    Returns:
        List[float]: a list containing f1-score per class
    """
    yp = np.argmax(y_pred, axis=1)
    n_classes = y_pred.shape[1]

    f2_scores = []
    for cl in range(n_classes):
        f2_scores.append(fbeta_score(y_true, yp, beta=2, average=None, labels=[cl])[0])

    return f2_scores


def per_class_f05score(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    """Per class F1 Score metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth
        y_pred (np.ndarray): numpy array containing the predictions

    Returns:
        List[float]: a list containing f1-score per class
    """
    yp = np.argmax(y_pred, axis=1)
    n_classes = y_pred.shape[1]

    f05_score = []
    for cl in range(n_classes):
        f05_score.append(
            fbeta_score(y_true, yp, beta=0.5, average=None, labels=[cl])[0]
        )

    return f05_score

##
# 24027
##


def fpr(y_true, y_pred):
    yp = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, yp)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)

    # # Overall accuracy
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    return FPR


def ppv(y_true, y_pred):
    yp = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, yp)
    FP = cm.sum(axis=0) - np.diag(cm)
    # FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    # TN = cm.sum() - (FP + FN + TP)

    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)

    # # Overall accuracy
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    return PPV


def npv(y_true, y_pred):
    yp = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, yp)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)

    # # Overall accuracy
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    return NPV


def tpr(y_true, y_pred):
    yp = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, yp)
    # FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    # TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)

    # # Overall accuracy
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    return TPR


def tnr(y_true, y_pred):
    yp = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, yp)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)

    # # Overall accuracy
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    return TNR


def fnr(y_true, y_pred):
    yp = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, yp)
    # FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    # TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)

    # # Overall accuracy
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    return FNR


def demographic_parity(y_true, y_pred: np.ndarray):
    yp = np.argmax(y_pred, axis=1)
    n_classes = y_pred.shape[1]
    dp = np.zeros(shape=n_classes, dtype=np.float32)
    for cl in range(n_classes):
        yp_cl = yp == cl
        positive_predictions = np.sum(yp_cl)
        total_instances = len(y_true)
        ppr = positive_predictions / total_instances
        # npr = 1 - ppr
        dp[cl] = ppr  # min(ppr, npr)

    return dp.tolist()


def matthews_correlation(y_true, y_pred):
    yp = np.argmax(y_pred, axis=1)
    return matthews_corrcoef(y_true, yp)


def balanced_accuracy(y_true, y_pred):
    yp = np.argmax(y_pred, axis=1)
    return balanced_accuracy_score(y_true, yp)


def generate_report(y_true, y_pred) -> None:
    confusion_matrix = confusionmatrix(y_true, y_pred)
    specificity_score = specificity(y_true, y_pred)
    accuracy_score = accuracy(y_true, y_pred)
    f1_score = f1score(y_true, y_pred)
    f05_score = f05score(y_true, y_pred)
    f2score_score = f2score(y_true, y_pred)
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)

    #Per class metrics

    per_class_specificity_score = per_class_specificity(y_true, y_pred)
    per_class_accuracy_score = per_class_accuracy(y_true, y_pred)
    per_class_precision_score = per_class_precision(y_true, y_pred)
    per_class_recall_score = per_class_recall(y_true, y_pred)
    per_class_f1score_score = per_class_f1score(y_true, y_pred)
    per_class_f2score_score = per_class_f2score(y_true, y_pred)
    per_class_f05score_score = per_class_f05score(y_true, y_pred)


    #cm = confusion_matrix(all_labels, all_predictions)

    print(f"Confusion Matrix: {confusion_matrix}")
    print(f"Validation Accuracy: {accuracy_score}")
    print(f"Precision: {precision_score}")
    print(f"Recall: {recall_score}")
    print(f"F1 Score: {f1_score}")
    print(f"specificity: {specificity_score}")
    print(f"f05_score: {f05_score}")
    print(f"f2score: {f2score_score}")

    print("Per class -------------------------------")
    print(f"per_class_specificity: {per_class_specificity_score}")
    print(f"per_class_accuracy: {per_class_accuracy_score}")
    print(f"per_class_precision: {per_class_precision_score}")
    print(f"per_class_recall: {per_class_recall_score}")
    print(f"per_class_f1score: {per_class_f1score_score}")
    print(f"per_class_f2score: {per_class_f2score_score}")
    print(f"per_class_f05score: {per_class_f05score_score}")