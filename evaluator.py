#!/usr/bin/env python
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from scipy.ndimage import label as connected_components
import pandas as pd

logger = logging.getLogger('satellite_anomaly')


def calculate_corrected_event_wise_fscore(y_true, y_pred, beta=0.5):
    """
    Calculate the corrected event-wise F-score as described in the Kotowski paper.

    This metric evaluates event detection rather than sample-by-sample detection and
    corrects for the issue of algorithms that detect anomalies everywhere.

    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        beta (float): Beta parameter for F-beta score

    Returns:
        tuple: (precision, recall, f_score)
    """
    # Identify events in the true and predicted labels
    # An event is a continuous segment of 1s
    true_events = identify_events(y_true)
    pred_events = identify_events(y_pred)

    # Count true negatives at the sample level (for TNR)
    tn = sum((y_true == 0) & (y_pred == 0))
    total_negatives = sum(y_true == 0)
    tnr = tn / total_negatives if total_negatives > 0 else 1.0

    # Calculate TP, FP, FN at the event level
    tp_e = 0
    for true_start, true_end in true_events:
        # An event is detected if any prediction in its range is 1
        detected = False
        for pred_start, pred_end in pred_events:
            # Check for overlap
            if max(true_start, pred_start) <= min(true_end, pred_end):
                detected = True
                break
        if detected:
            tp_e += 1

    fn_e = len(true_events) - tp_e
    fp_e = len(pred_events) - tp_e

    # Calculate corrected event-wise precision, recall, and F-score
    if tp_e + fp_e > 0:
        prec_e_corr = (tp_e / (tp_e + fp_e)) * tnr
    else:
        prec_e_corr = 0

    if tp_e + fn_e > 0:
        rec_e = tp_e / (tp_e + fn_e)
    else:
        rec_e = 0

    if prec_e_corr + rec_e > 0:
        f_beta_score = (1 + beta ** 2) * (prec_e_corr * rec_e) / ((beta ** 2 * prec_e_corr) + rec_e)
    else:
        f_beta_score = 0

    return prec_e_corr, rec_e, f_beta_score


def calculate_event_wise_alarming_precision(y_true, y_pred):
    """
    Calculate the event-wise alarming precision as described in the Kotowski paper.

    This metric evaluates if the algorithm generates multiple alarms for the same anomaly.

    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels

    Returns:
        float: Alarming precision
    """
    # Identify events in the true and predicted labels
    true_events = identify_events(y_true)
    pred_events = identify_events(y_pred)

    # Count true positives at the event level
    tp_e = 0
    # Count redundant alarms
    redundant_alarms = 0

    # Track which predicted events have been matched
    matched_preds = set()

    for i, (true_start, true_end) in enumerate(true_events):
        # Find all predicted events that overlap with this true event
        for j, (pred_start, pred_end) in enumerate(pred_events):
            # Check for overlap
            if max(true_start, pred_start) <= min(true_end, pred_end):
                if j not in matched_preds:
                    # First match is a true positive
                    tp_e += 1
                    matched_preds.add(j)
                else:
                    # Additional matches are redundant alarms
                    redundant_alarms += 1

    # Calculate alarming precision
    if tp_e + redundant_alarms > 0:
        alarm_precision = tp_e / (tp_e + redundant_alarms)
    else:
        alarm_precision = 1.0  # Perfect score if no alarms

    return alarm_precision


def calculate_adtqc(y_true, y_pred, timestamps=None):
    """
    Calculate the Anomaly Detection Timing Quality Curve (ADTQC) score.

    This metric evaluates how accurately the algorithm identifies the start time of anomalies.

    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        timestamps (list): List of timestamps for each prediction

    Returns:
        tuple: (adtqc_score, after_ratio)
    """
    # Identify events in the true and predicted labels
    true_events = identify_events(y_true)
    pred_events = identify_events(y_pred)

    if not true_events or not pred_events:
        return 0.0, 0.0

    # If timestamps provided, convert events to timestamp indices
    use_timestamps = timestamps is not None and len(timestamps) == len(y_true)

    quality_scores = []
    after_count = 0

    for true_start, true_end in true_events:
        # Find best matching predicted event
        best_quality = 0
        best_is_after = False

        for pred_start, pred_end in pred_events:
            # Skip if no overlap at all
            if pred_end < true_start or pred_start > true_end:
                continue

            # Calculate timing difference
            time_diff = pred_start - true_start

            # Convert to time units if timestamps available
            if use_timestamps:
                if isinstance(timestamps[0], pd.Timestamp):
                    # Convert to seconds for timestamp objects
                    true_time = timestamps[true_start]
                    pred_time = timestamps[pred_start]
                    time_diff = (pred_time - true_time).total_seconds()
                else:
                    # Simple index difference if not using timestamps
                    time_diff = pred_start - true_start

            # Calculate ADTQC quality
            event_length = true_end - true_start
            quality = calculate_timing_quality(time_diff, event_length)

            if quality > best_quality:
                best_quality = quality
                best_is_after = time_diff >= 0

        if best_quality > 0:
            quality_scores.append(best_quality)
            if best_is_after:
                after_count += 1

    # Calculate average ADTQC score and after ratio
    if quality_scores:
        adtqc_score = sum(quality_scores) / len(quality_scores)
        after_ratio = after_count / len(quality_scores)
    else:
        adtqc_score = 0.0
        after_ratio = 0.0

    return adtqc_score, after_ratio


def calculate_timing_quality(time_diff, event_length):
    """
    Calculate the timing quality value based on the ADTQC function.

    Args:
        time_diff (float): Time difference between prediction and true anomaly start
        event_length (float): Length of the true anomaly event

    Returns:
        float: Timing quality score
    """
    # Parameters for the ADTQC curve
    alpha = min(event_length, abs(event_length))
    beta = event_length

    # Calculate quality based on timing difference
    if time_diff <= -alpha:
        # Too early
        return 0.0
    elif -alpha < time_diff <= 0:
        # Early but within acceptable range
        return ((time_diff + alpha) / alpha) ** 2
    elif 0 < time_diff < beta:
        # Late but within acceptable range
        return 1.0 / (1.0 + (time_diff / (beta - time_diff)) ** 2)
    else:
        # Too late
        return 0.0


def identify_events(binary_sequence):
    """
    Identify continuous events in a binary sequence.

    Args:
        binary_sequence (numpy.ndarray): Array of binary values

    Returns:
        list: List of (start, end) tuples for each event
    """
    # Find connected components
    labeled_array, num_features = connected_components(binary_sequence)

    events = []
    for i in range(1, num_features + 1):
        # Find indices where this label appears
        indices = np.where(labeled_array == i)[0]
        if len(indices) > 0:
            events.append((indices[0], indices[-1]))

    return events


def evaluate_predictions(y_true, y_pred, group_name=None, beta=0.5, output_dir=None, timestamps=None):
    """
    Evaluate anomaly detection performance

    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        group_name (str): Name of channel group
        beta (float): Beta parameter for F-beta score
        output_dir (str): Directory for output visualizations
        timestamps (list): List of timestamps for each prediction

    Returns:
        dict: Evaluation results
    """
    prefix = f"Group {group_name}: " if group_name else ""
    logger.info(f"{prefix}Evaluating performance")

    # Basic metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # F-beta score (beta=0.5 emphasizes precision)
    if precision + recall > 0:
        f_beta = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
    else:
        f_beta = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Advanced metrics from the Kotowski paper
    # Corrected event-wise F-score
    event_prec, event_rec, event_f_beta = calculate_corrected_event_wise_fscore(y_true, y_pred, beta)

    # Event-wise alarming precision
    alarming_precision = calculate_event_wise_alarming_precision(y_true, y_pred)

    # ADTQC score
    adtqc_score, after_ratio = calculate_adtqc(y_true, y_pred, timestamps)

    # Log results
    logger.info(f"{prefix}Sample-level Precision: {precision:.4f}")
    logger.info(f"{prefix}Sample-level Recall: {recall:.4f}")
    logger.info(f"{prefix}Sample-level F{beta}-score: {f_beta:.4f}")
    logger.info(f"{prefix}Event-wise Precision: {event_prec:.4f}")
    logger.info(f"{prefix}Event-wise Recall: {event_rec:.4f}")
    logger.info(f"{prefix}Event-wise F{beta}-score: {event_f_beta:.4f}")
    logger.info(f"{prefix}Alarming Precision: {alarming_precision:.4f}")
    logger.info(f"{prefix}ADTQC Score: {adtqc_score:.4f} (After Ratio: {after_ratio:.4f})")
    logger.info(f"{prefix}Confusion matrix: \n{cm}")

    # Create results dictionary
    results = {
        'precision': float(precision),
        'recall': float(recall),
        'f_beta': float(f_beta),
        'event_precision': float(event_prec),
        'event_recall': float(event_rec),
        'event_f_beta': float(event_f_beta),
        'alarming_precision': float(alarming_precision),
        'adtqc_score': float(adtqc_score),
        'adtqc_after_ratio': float(after_ratio),
        'confusion_matrix': cm.tolist()
    }

    # Create confusion matrix visualization
    if group_name is not None and output_dir is not None:
        visualize_confusion_matrix(cm, group_name, output_dir)

    return results


def visualize_confusion_matrix(cm, group_name, output_dir):
    """
    Create and save confusion matrix visualization

    Args:
        cm (numpy.ndarray): Confusion matrix
        group_name (str): Name of channel group
        output_dir (str): Directory for output visualizations
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(cm, cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - Group {group_name}')
    ax.xaxis.set_label_position('top')

    # Save visualization
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, f'confusion_matrix_group_{group_name}.png'))
    plt.close()


def combine_group_predictions(all_results, output_dir=None, beta=0.5):
    """
    Combine and evaluate results from all groups

    Args:
        all_results (dict): Results from all groups
        output_dir (str): Directory for output visualizations
        beta (float): Beta parameter for F-beta score

    Returns:
        dict: Combined evaluation results
    """
    logger.info("Combining predictions from all groups")

    # Combine all true and predicted values
    all_y_true = []
    all_y_pred = []

    for group, result in all_results.items():
        if 'y_true' in result and 'y_pred' in result:
            all_y_true.extend(result['y_true'])
            all_y_pred.extend(result['y_pred'])

    # Convert to numpy arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Only evaluate if we have data
    if len(all_y_true) > 0:
        # Calculate basic metrics
        precision = precision_score(all_y_true, all_y_pred, zero_division=0)
        recall = recall_score(all_y_true, all_y_pred, zero_division=0)

        # F-beta score
        if precision + recall > 0:
            f_beta = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
        else:
            f_beta = 0.0

        # Confusion matrix
        cm = confusion_matrix(all_y_true, all_y_pred)

        # Advanced metrics from the Kotowski paper
        # Corrected event-wise F-score
        event_prec, event_rec, event_f_beta = calculate_corrected_event_wise_fscore(all_y_true, all_y_pred, beta)

        # Event-wise alarming precision
        alarming_precision = calculate_event_wise_alarming_precision(all_y_true, all_y_pred)

        # ADTQC score (without timestamps in combined evaluation)
        adtqc_score, after_ratio = calculate_adtqc(all_y_true, all_y_pred)

        # Log results
        logger.info(f"OVERALL Sample-level Precision: {precision:.4f}")
        logger.info(f"OVERALL Sample-level Recall: {recall:.4f}")
        logger.info(f"OVERALL Sample-level F{beta}-score: {f_beta:.4f}")
        logger.info(f"OVERALL Event-wise Precision: {event_prec:.4f}")
        logger.info(f"OVERALL Event-wise Recall: {event_rec:.4f}")
        logger.info(f"OVERALL Event-wise F{beta}-score: {event_f_beta:.4f}")
        logger.info(f"OVERALL Alarming Precision: {alarming_precision:.4f}")
        logger.info(f"OVERALL ADTQC Score: {adtqc_score:.4f} (After Ratio: {after_ratio:.4f})")
        logger.info(f"OVERALL Confusion matrix: \n{cm}")

        # Create visualization
        if output_dir:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.matshow(cm, cmap=plt.cm.Blues)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center')

            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Combined Confusion Matrix (All Groups)')
            ax.xaxis.set_label_position('top')

            # Save visualization
            plt.savefig(os.path.join(vis_dir, 'confusion_matrix_overall.png'))
            plt.close()

        # Return combined results
        return {
            'precision': precision,
            'recall': recall,
            'f_beta': f_beta,
            'event_precision': float(event_prec),
            'event_recall': float(event_rec),
            'event_f_beta': float(event_f_beta),
            'alarming_precision': float(alarming_precision),
            'adtqc_score': float(adtqc_score),
            'adtqc_after_ratio': float(after_ratio),
            'confusion_matrix': cm.tolist()
        }
    else:
        logger.warning("No combined predictions available for evaluation")
        return None