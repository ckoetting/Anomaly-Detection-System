import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter

def plot_anomaly_scores(timestamps, scores, predictions=None, true_anomalies=None, threshold=None, figsize=(15, 8)):
    """
    Plot anomaly scores with detected anomalies
    
    Args:
        timestamps (array-like): Timestamps for the data points
        scores (array-like): Anomaly scores
        predictions (array-like, optional): Model predictions (1 = anomaly, 0 = normal)
        true_anomalies (array-like, optional): True anomaly labels
        threshold (float, optional): Anomaly threshold
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot anomaly scores
    plt.plot(timestamps, scores, 'b-', alpha=0.6, label='Anomaly Score')
    
    # Plot threshold if provided
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.3f})')
    
    # Highlight detected anomalies if provided
    if predictions is not None:
        anomaly_idx = np.where(predictions == 1)[0]
        plt.scatter(
            [timestamps[i] for i in anomaly_idx], 
            [scores[i] for i in anomaly_idx],
            color='red', label='Detected Anomalies', s=50, zorder=5
        )
    
    # Highlight true anomalies if provided
    if true_anomalies is not None:
        true_idx = np.where(true_anomalies == 1)[0]
        plt.scatter(
            [timestamps[i] for i in true_idx], 
            [scores[i] for i in true_idx],
            color='green', label='True Anomalies', marker='x', s=50, zorder=4
        )
    
    plt.title('Anomaly Detection Results')
    plt.xlabel('Timestamp')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format date on x-axis if timestamps are datetime objects
    if isinstance(timestamps[0], pd.Timestamp):
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
    
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve(fpr, tpr, roc_auc, figsize=(10, 6)):
    """
    Plot ROC curve for the anomaly detection model
    
    Args:
        fpr (array-like): False positive rates
        tpr (array-like): True positive rates
        roc_auc (float): Area under ROC curve
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_precision_recall_curve(precision, recall, pr_auc, figsize=(10, 6)):
    """
    Plot precision-recall curve for the anomaly detection model
    
    Args:
        precision (array-like): Precision values
        recall (array-like): Recall values
        pr_auc (float): Area under precision-recall curve
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_confusion_matrix(conf_matrix, figsize=(8, 6)):
    """
    Plot confusion matrix for the anomaly detection results
    
    Args:
        conf_matrix (array-like): Confusion matrix (2x2)
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=['Normal', 'Anomaly'],
        yticklabels=['Normal', 'Anomaly']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt.gcf()
