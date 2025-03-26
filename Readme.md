# SatGuard Pro: Satellite Anomaly Detection System

A comprehensive system for detecting anomalies in satellite telemetry data using machine learning techniques. This repository contains the core Python modules for the anomaly detection pipeline, designed to work with ESA satellite telemetry data.

## Overview

SatGuard Pro implements an improved approach to satellite anomaly detection based on Isolation Forest with specialized evaluation metrics. The system is designed to:

1. Process and prepare satellite telemetry data
2. Train anomaly detection models for different channel groups
3. Evaluate model performance using advanced metrics
4. Visualize results for easier interpretation

This anomaly detection system includes implementations of metrics from recent research in the field, including corrected event-wise F-scores, alarming precision, and Anomaly Detection Timing Quality Curve (ADTQC) scores.

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy

You can install the required dependencies using:

```bash
pip install numpy pandas scikit-learn matplotlib scipy
```

## Data Preparation

Before using this system, you need to download and preprocess the ESA satellite telemetry data. Please follow the instructions in the [ESA-ADB repository](https://github.com/kplabs-pl/ESA-ADB/tree/main) to download and prepare the necessary data files.

The system expects the following files:
- `data/21_months.train.csv` - Telemetry data
- `data/labels.csv` - Anomaly labels
- `data/channels.csv` - Channel metadata
- `data/anomaly_types.csv` - Anomaly type definitions

## Core Modules

### data_loader.py

This module handles loading and preprocessing of satellite telemetry data.

Key functions:
- `load_channel_metadata()`: Loads channel grouping information
- `load_data()`: Loads telemetry data from CSV files
- `add_anomaly_labels()`: Adds anomaly labels to the telemetry data
- `preprocess_data()`: Performs resampling and standardization
- `implement_temporal_split()`: Splits data into train, validation, and test sets
- `create_windows_for_group()`: Creates sliding windows for feature extraction

### model.py

This module contains functions for training anomaly detection models and making predictions.

Key functions:
- `train_model()`: Trains an Isolation Forest model
- `predict_anomalies()`: Makes predictions with improved processing, including smoothing and minimum duration filtering

### evaluator.py

This module provides comprehensive evaluation metrics for anomaly detection performance.

Key functions:
- `evaluate_predictions()`: Evaluates predictions using multiple metrics
- `calculate_corrected_event_wise_fscore()`: Calculates event-based F-score with correction for false alarms
- `calculate_event_wise_alarming_precision()`: Evaluates if the algorithm generates multiple alarms for the same anomaly
- `calculate_adtqc()`: Calculates the Anomaly Detection Timing Quality Curve score
- `combine_group_predictions()`: Combines and evaluates results from all groups

### results_visuals.py

This module provides visualization tools for model performance evaluation.

It creates a bubble chart to compare the performance of different channel groups, with:
- X-axis representing Precision
- Y-axis representing Recall
- Bubble size representing the number of channels in each group
- Color representing the Event F0.5 score

## Example Usage

Here's a basic workflow to use the system:

```python
import os
from data_loader import (
    load_channel_metadata, load_data, add_anomaly_labels,
    preprocess_data, implement_temporal_split, create_windows_for_group
)
from model import train_model, predict_anomalies
from evaluator import evaluate_predictions, combine_group_predictions

# Setup
data_dir = "data"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# 1. Load channel metadata
target_channels, channel_groups, group_to_channels = load_channel_metadata(
    os.path.join(data_dir, "channels.csv")
)

# 2. Load telemetry data
df = load_data(
    os.path.join(data_dir, "21_months.train.csv"),
    target_channels=target_channels,
    mission1_only=True
)

# 3. Add anomaly labels
df = add_anomaly_labels(
    df,
    os.path.join(data_dir, "labels.csv"),
    os.path.join(data_dir, "anomaly_types.csv")
)

# 4. Preprocess data
df = preprocess_data(df, resample_freq="30s")

# 5. Split data
train_df, validation_df, test_df = implement_temporal_split(df)

# 6. Train and evaluate for each group
all_results = {}

for group, channels in group_to_channels.items():
    print(f"Processing group {group}")
    
    # Create windows
    X_train, y_train, _ = create_windows_for_group(
        train_df, channels, window_size=256, step_size=128
    )
    
    X_test, y_test, test_timestamps = create_windows_for_group(
        test_df, channels, window_size=256, step_size=128
    )
    
    # Skip if no data
    if X_train is None or len(X_train) == 0 or X_test is None or len(X_test) == 0:
        print(f"Skipping group {group} - no valid data")
        continue
    
    # Train model
    model = train_model(X_train, y_train, n_estimators=500)
    
    # Make predictions
    y_pred, scores = predict_anomalies(
        model,
        X_test,
        threshold_percentile=98,
        smoothing_window=10,
        min_anomaly_duration=25
    )
    
    # Evaluate
    results = evaluate_predictions(y_test, y_pred, group, beta=0.5)
    
    # Store results
    all_results[group] = {
        'results': results,
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'channels': channels
    }

# 7. Combine and evaluate overall results
overall_results = combine_group_predictions(all_results)
```

## Evaluation Metrics

The system implements several advanced evaluation metrics:

1. **Corrected Event-wise F-score**: Evaluates event detection rather than sample-by-sample detection, correcting for the issue of algorithms that detect anomalies everywhere.

2. **Event-wise Alarming Precision**: Measures if the algorithm generates multiple alarms for the same anomaly event.

3. **Anomaly Detection Timing Quality Curve (ADTQC)**: Evaluates how accurately the algorithm identifies the start time of anomalies.

4. **Standard metrics**: Precision, recall, and F-beta score (with beta=0.5, emphasizing precision).

## Visualization

You can visualize the performance comparison between different channel groups using the `results_visuals.py` script:

```python
import matplotlib.pyplot as plt
from results_visuals import group_results  # This will contain your results

# The plot will show a bubble chart comparing groups by precision, recall, and F-score
plt.show()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on research and methodologies from the ESA satellite anomaly detection benchmark
- Evaluation metrics based on research by Kotowski et al.
