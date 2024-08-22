import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    precision_recall_curve,
    auc,
    fbeta_score,
    average_precision_score
)


def get_predictions(dataframe):

    all_labels = dataframe['true_class'].values
    all_predictions = dataframe['pred_class'].values
    all_probs = dataframe['prob'].values
    return all_labels, all_predictions, all_probs


# Taken from scikitplot
def aucrc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under Cumulative Gains Curve.

    Args:
        y_true (np.ndarray): ground truth
        y_score (np.ndarray): predicted probabilities/scores

    Returns:
        float: area under the RC curve
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    # make y_true a boolean vector
    y_true = y_true == 1
    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)
    value_range = np.arange(start=1, stop=len(y_true) + 1)
    gains = gains / float(np.sum(y_true))
    percentages = value_range / float(len(y_true))
    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])
    return auc(percentages, gains)


def get_metrics(all_labels, all_predictions, all_probs, classes):
    # Calculate metrics
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()

    # AUROC
    auroc = roc_auc_score(all_labels, all_probs, labels=classes)
    aucrc_score = aucrc(all_labels, all_probs)

    # AUPRC
    #precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
    #auprc = auc(recall_vals, precision_vals)
    auprc = average_precision_score(all_labels, all_probs)

    # F-beta scores
    f_beta_0_5 = fbeta_score(all_labels, all_predictions, beta=0.5)
    f_beta_2 = fbeta_score(all_labels, all_predictions, beta=2)

    # False Discovery Rate (FDR)
    fdr = fp / (fp + tp)

    # False Negative Rate (FNR)
    fnr = fn / (fn + tp)

    # False Omission Rate (FOR)
    for_ = fn / (fn + tn)

    # False Positive Rate (FPR)
    fpr = fp / (fp + tn)

    # Negative Predictive Value (NPV)
    npv = tn / (tn + fn)

    # Negative Likelihood Ratio (NLR)
    nlr = fnr / (tn / (tn + fp))

    # Positive Likelihood Ratio (PLR)
    plr = recall / fpr

    # Prevalence
    prevalence = (tp + fn) / (tp + tn + fp + fn)

    # True Negative Rate (TNR)
    tnr = tn / (tn + fp)


    accuracy = accuracy_score(all_labels,all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot().figure_.savefig('confusion_matrix.png')
    print(f"Confusion Matrix: {cm}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"auroc score: {auroc:.4f}")
    print(f"aucrc score: {aucrc_score:.4f}")
    

    print(f"auprc: {auprc:.4f}")
    print(f"F-beta scores _0_5: {f_beta_0_5:.4f}")
    print(f"F-beta scores _2: {f_beta_2:.4f}")
    print(f"False Discovery Rate (FDR): {fdr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"False Omission Rate (FOR): {for_:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"Negative Predictive Value (NPV): {npv:.4f}")
    print(f"Negative Likelihood Ratio (NLR): {nlr:.4f}")
    print(f"Positive Likelihood Ratio (PLR): {plr:.4f}")
    print(f"prevalence: {prevalence:.4f}")
    print(f"True Negative Rate (TNR): {tnr:.4f}")


if __name__ == "__main__":
    classes = ['brain', 'chest']
    csv_path = 'docker_results_probabilities.csv'
    predictions_df = pd.read_csv(csv_path)
    true_labels, pred_labels, probs = get_predictions(predictions_df)
    get_metrics(true_labels, pred_labels, probs, classes)

