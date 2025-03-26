#!/usr/bin/env python
import argparse
import os
import json
import logging
import time
import pickle
from datetime import timedelta

# Import from other modules
from data_loader import (
    load_channel_metadata, load_data, add_anomaly_labels,
    preprocess_data, implement_temporal_split, create_windows_for_group
)
from model import train_model, predict_anomalies
from evaluator import evaluate_predictions, combine_group_predictions


def setup_logging(output_dir):
    """Simple logging setup"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'satellite_anomaly.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('satellite_anomaly')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Improved satellite anomaly detection")

    # Data paths
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--train_file", type=str, required=True, help="Training file name")
    parser.add_argument("--labels_file", type=str, default="labels.csv", help="Labels file name")
    parser.add_argument("--channels_file", type=str, default="channels.csv", help="Channels file name")
    parser.add_argument("--anomaly_types_file", type=str, default="anomaly_types.csv", help="Anomaly types file name")

    # Processing options
    parser.add_argument("--all_channels", action="store_true", help="Use all channels instead of only target ones")
    parser.add_argument("--resample_freq", type=str, default="30s", help="Resampling frequency")
    parser.add_argument("--window_size", type=int, default=256, help="Window size")
    parser.add_argument("--step_size", type=int, default=128, help="Step size")

    # Model options
    parser.add_argument("--n_estimators", type=int, default=500, help="Number of estimators")
    parser.add_argument("--threshold_percentile", type=float, default=98, help="Threshold percentile")
    parser.add_argument("--smoothing_window", type=int, default=10, help="Smoothing window size")
    parser.add_argument("--min_anomaly_duration", type=int, default=25, help="Minimum anomaly duration")

    # Output options
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting improved satellite anomaly detection")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)

    # 1. Load channel metadata
    target_channels, channel_groups, group_to_channels = load_channel_metadata(
        os.path.join(args.data_dir, args.channels_file)
    )

    # 2. Load raw data (all channels if using all_channels)
    df = load_data(
        os.path.join(args.data_dir, args.train_file),
        target_channels=None if args.all_channels else target_channels,
        mission1_only=True
    )

    # 3. Add anomaly labels
    df = add_anomaly_labels(
        df,
        os.path.join(args.data_dir, args.labels_file),
        os.path.join(args.data_dir, args.anomaly_types_file)
    )

    # 4. Preprocess data
    df = preprocess_data(df, args.resample_freq)

    # 5. Split data
    train_df, validation_df, test_df = implement_temporal_split(df)

    # 6. Process each group separately
    all_results = {}

    for group, channels in group_to_channels.items():
        logger.info(f"\n{'=' * 80}\nProcessing channel group {group}: {channels}\n{'=' * 80}")

        # Create windows for this group
        X_train, y_train, _ = create_windows_for_group(
            train_df, channels, args.window_size, args.step_size
        )

        # Skip if no data
        if X_train is None or len(X_train) == 0:
            logger.warning(f"Skipping group {group} - no valid data")
            continue

        X_test, y_test, test_timestamps = create_windows_for_group(
            test_df, channels, args.window_size, args.step_size
        )

        # Skip if no test data
        if X_test is None or len(X_test) == 0:
            logger.warning(f"Skipping group {group} - no valid test data")
            continue

        # Train model
        model = train_model(X_train, y_train, args.n_estimators)

        # Make predictions
        y_pred, scores = predict_anomalies(
            model,
            X_test,
            threshold_percentile=args.threshold_percentile,
            smoothing_window=args.smoothing_window,
            min_anomaly_duration=args.min_anomaly_duration
        )

        # Evaluate performance
        results = evaluate_predictions(y_test, y_pred, group, beta=0.5, output_dir=args.output_dir,
                                       timestamps=test_timestamps)

        # Store results
        all_results[group] = {
            'results': results,
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'channels': channels
        }

        # Save model
        model_path = os.path.join(args.output_dir, 'models', f'model_group_{group}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    # 7. Combine and evaluate overall results
    overall_results = combine_group_predictions(all_results, args.output_dir)

    # 8. Save all results
    with open(os.path.join(args.output_dir, 'group_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for group, result in all_results.items():
            serializable_results[str(group)] = {
                'results': result['results'],
                'channels': result['channels']
            }
        json.dump(serializable_results, f, indent=2)

    if overall_results:
        with open(os.path.join(args.output_dir, 'overall_results.json'), 'w') as f:
            json.dump(overall_results, f, indent=2)

    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    logger.info("Done!")


if __name__ == "__main__":
    main()