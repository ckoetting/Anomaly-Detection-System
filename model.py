#!/usr/bin/env python
import numpy as np
import logging
from sklearn.ensemble import IsolationForest

logger = logging.getLogger('satellite_anomaly')


def train_model(X, y, n_estimators=100, contamination=None):
    """
    Train Isolation Forest model

    Args:
        X (numpy.ndarray): Feature array
        y (numpy.ndarray): Label array
        n_estimators (int): Number of estimators for Isolation Forest
        contamination (float): Contamination parameter for Isolation Forest

    Returns:
        object: Trained Isolation Forest model
    """
    # Set contamination based on actual anomaly rate if not provided
    if contamination is None:
        actual_contamination = sum(y) / len(y)
        # Use at least 0.02 (2%) contamination to avoid being too conservative
        contamination = max(0.02, actual_contamination)

    logger.info(f"Training Isolation Forest with contamination={contamination:.4f}")

    # Train on normal data only
    if sum(y) > 0:
        X_normal = X[y == 0]
        logger.info(f"Training on {len(X_normal)} normal samples")
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_normal)
    else:
        logger.info("No anomalies found, training on all data")
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X)

    return model


def predict_anomalies(model, X, threshold_percentile=90, smoothing_window=10, min_anomaly_duration=5):
    """
    Predict anomalies with improved processing

    Args:
        model (object): Trained Isolation Forest model
        X (numpy.ndarray): Feature array
        threshold_percentile (float): Percentile for anomaly threshold
        smoothing_window (int): Window size for smoothing anomaly scores
        min_anomaly_duration (int): Minimum duration for anomalies

    Returns:
        tuple: (predictions, smoothed_scores) containing binary predictions and smoothed anomaly scores
    """
    logger.info("Predicting anomalies")

    # Get anomaly scores (higher = more anomalous)
    raw_scores = -model.decision_function(X)

    # Log score statistics
    logger.info(
        f"Score statistics: min={raw_scores.min():.4f}, max={raw_scores.max():.4f}, mean={raw_scores.mean():.4f}")

    # Apply smoothing
    if smoothing_window > 1:
        logger.info(f"Applying smoothing with window={smoothing_window}")
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed_scores = np.convolve(raw_scores, kernel, mode='same')
    else:
        smoothed_scores = raw_scores

    # Apply threshold (lower percentile for better precision)
    logger.info(f"Using {threshold_percentile}th percentile as threshold")
    threshold = np.percentile(smoothed_scores, threshold_percentile)
    logger.info(f"Threshold value: {threshold:.4f}")

    # Get binary predictions
    predictions = (smoothed_scores > threshold).astype(int)

    # Prune short anomalies
    if min_anomaly_duration > 1:
        logger.info(f"Pruning anomalies shorter than {min_anomaly_duration}")
        pruned = predictions.copy()

        # Find consecutive anomalies
        i = 0
        while i < len(predictions):
            if predictions[i] == 1:
                # Start of anomaly
                start = i
                while i < len(predictions) and predictions[i] == 1:
                    i += 1
                end = i - 1

                # Check duration
                if end - start + 1 < min_anomaly_duration:
                    pruned[start:end + 1] = 0
            else:
                i += 1

        predictions = pruned

    anomaly_count = np.sum(predictions)
    logger.info(f"Final prediction: {anomaly_count} anomalies ({anomaly_count / len(predictions) * 100:.2f}%)")

    return predictions, smoothed_scores