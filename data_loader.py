#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

logger = logging.getLogger('satellite_anomaly')


def load_channel_metadata(channels_path):
    """
    Load channel metadata to get grouping information

    Args:
        channels_path (str): Path to channels CSV file

    Returns:
        tuple: (target_channels, channel_groups, groups) containing target channel names,
               channel to group mapping, and group to channels mapping
    """
    logger.info(f"Loading channel metadata from {channels_path}")
    channels_df = pd.read_csv(channels_path)

    # Print column names and first few rows to debug
    logger.info(f"Channel CSV columns: {channels_df.columns.tolist()}")
    logger.info(f"First few rows of channel data:\n{channels_df.head()}")

    # Print unique values in the Target column
    if 'Target' in channels_df.columns:
        logger.info(f"Unique values in Target column: {channels_df['Target'].unique()}")

    # Identify target channels more flexibly
    if 'Target' in channels_df.columns:
        # Try different possible values for target channels
        target_indicators = ['Yes', 'YES', 'yes', 'true', 'TRUE', 'True', '1', 'y', 'Y']
        mask = channels_df['Target'].astype(str).isin(target_indicators)
        target_channels = channels_df[mask]['Channel'].tolist()
    else:
        # If no Target column, assume all channels are targets
        logger.warning("No 'Target' column found, assuming all channels are targets")
        target_channels = channels_df['Channel'].tolist()

    # Create channel-to-group mapping
    channel_groups = {}
    for _, row in channels_df.iterrows():
        channel = row['Channel']
        if 'Group' in channels_df.columns:
            group = row['Group']
            channel_groups[channel] = group
        else:
            # If no Group column, use default grouping
            logger.warning("No 'Group' column found, using default grouping")
            channel_groups[channel] = 1

    # Create group-to-channels mapping
    groups = defaultdict(list)
    for _, row in channels_df.iterrows():
        channel = row['Channel']
        if 'Group' in channels_df.columns:
            group = row['Group']
        else:
            group = 1

        # Check if this is a target channel
        if 'Target' in channels_df.columns:
            is_target = str(row['Target']).lower() in [str(x).lower() for x in target_indicators]
        else:
            is_target = True

        if is_target:  # Only include target channels
            groups[group].append(channel)

    logger.info(f"Found {len(groups)} channel groups with {len(target_channels)} target channels")

    # If no groups were found, create a single group with all channels
    if len(groups) == 0:
        logger.warning("No valid groups found. Creating a single group with all channels.")
        for channel in target_channels:
            groups[1].append(channel)
        for channel in target_channels:
            channel_groups[channel] = 1

    return target_channels, channel_groups, groups


def load_data(data_path, target_channels=None, mission1_only=True):
    """
    Load telemetry data with support for all channels

    Args:
        data_path (str): Path to data CSV file
        target_channels (list): List of target channel names to keep
        mission1_only (bool): If True, only keep Mission1 channels

    Returns:
        pandas.DataFrame: Loaded telemetry data
    """
    logger.info(f"Loading data from {data_path}")

    # Load raw data
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter for Mission1 channels if requested
    if mission1_only:
        mission_channels = [col for col in df.columns if col.startswith('channel_')]
        cols_to_keep = ['timestamp'] + mission_channels
        df = df[cols_to_keep]

    # Select only target channels if provided
    if target_channels:
        cols_to_keep = ['timestamp'] + [col for col in target_channels if col in df.columns]
        logger.info(f"Selecting {len(cols_to_keep) - 1} target channels")
        df = df[cols_to_keep]

    return df


def add_anomaly_labels(df, labels_path, anomaly_types_path):
    """
    Add anomaly labels with case-insensitive channel matching

    Args:
        df (pandas.DataFrame): Telemetry data
        labels_path (str): Path to labels CSV file
        anomaly_types_path (str): Path to anomaly types CSV file

    Returns:
        pandas.DataFrame: Telemetry data with anomaly labels added
    """
    logger.info("Adding anomaly labels to data")

    # Load labels and anomaly types
    labels = pd.read_csv(labels_path)
    anomaly_types = pd.read_csv(anomaly_types_path)

    # Create mapping from ID to category
    id_to_category = {}
    for _, row in anomaly_types.iterrows():
        id_to_category[row['ID']] = row['Category'].lower() if isinstance(row['Category'], str) else "anomaly"

    # Create all anomaly columns at once to avoid fragmentation
    new_columns = {}
    for channel in df.columns:
        if channel != 'timestamp':
            new_columns[f'is_anomaly_{channel}'] = np.zeros(len(df))
            new_columns[f'is_rare_event_{channel}'] = np.zeros(len(df))

    # Add all columns at once using a new DataFrame and concat
    anomaly_df = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, anomaly_df], axis=1)

    # Get actual channel names in lowercase for case-insensitive matching
    df_channels = {col.lower(): col for col in df.columns if col != 'timestamp'}

    # Process each label
    labels_applied = 0
    skipped = 0

    for _, row in labels.iterrows():
        # Handle potential case sensitivity in channel names
        channel = row['Channel']
        channel_lower = channel.lower()

        # Skip if channel not in dataframe
        if channel_lower not in df_channels:
            skipped += 1
            continue

        # Use the actual case-sensitive channel name
        actual_channel = df_channels[channel_lower]

        # Get event category
        event_id = row['ID']
        category = id_to_category.get(event_id, "anomaly")

        # Skip communication gaps
        if category.lower() == "communication gap":
            continue

        # Get timestamps and label column - ENSURE TIMEZONE CONSISTENCY
        start_time = pd.to_datetime(row['StartTime'])
        end_time = pd.to_datetime(row['EndTime'])

        # Make timestamps timezone-naive to match DataFrame (remove timezone info)
        if start_time.tzinfo is not None:
            start_time = start_time.replace(tzinfo=None)
        if end_time.tzinfo is not None:
            end_time = end_time.replace(tzinfo=None)

        # Determine which column to update based on category
        if category.lower() == "anomaly":
            col_name = f'is_anomaly_{actual_channel}'
        elif category.lower() in ["rare nominal event", "rare event"]:
            col_name = f'is_rare_event_{actual_channel}'
        else:
            continue

        # Apply label
        mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
        df.loc[mask, col_name] = 1
        labels_applied += 1

    logger.info(f"Applied {labels_applied} labels, skipped {skipped} labels")

    # Count anomalies
    anomaly_cols = [c for c in df.columns if c.startswith('is_anomaly_')]
    if anomaly_cols:
        total_anomalies = 0
        for col in anomaly_cols:
            count = df[col].sum()
            if count > 0:
                logger.info(f"{col}: {count} anomalies ({count / len(df) * 100:.4f}%)")
                total_anomalies += count
        logger.info(f"Total anomalies across all channels: {total_anomalies}")

    return df


def preprocess_data(df, resample_freq="30s"):
    """
    Preprocess telemetry data - resample and standardize

    Args:
        df (pandas.DataFrame): Telemetry data
        resample_freq (str): Resampling frequency

    Returns:
        pandas.DataFrame: Preprocessed telemetry data
    """
    logger.info(f"Preprocessing data with resample_freq={resample_freq}")

    # Resample data
    df = df.set_index('timestamp')

    # Separate features and labels
    feature_cols = [col for col in df.columns if
                    not col.startswith('is_anomaly_') and not col.startswith('is_rare_event_')]
    label_cols = [col for col in df.columns if col.startswith('is_anomaly_') or col.startswith('is_rare_event_')]

    # Resample features (last value) and labels (max value)
    df_resampled = pd.concat([
        df[feature_cols].resample(resample_freq).ffill(),
        df[label_cols].resample(resample_freq).max()
    ], axis=1).reset_index()

    # Standardize features (z-score normalization)
    for col in feature_cols:
        # Get mean and std from non-anomalous data
        anomaly_mask = df_resampled[[c for c in label_cols if c.startswith('is_anomaly_')]].sum(axis=1) > 0
        normal_data = df_resampled.loc[~anomaly_mask, col]

        mean_val = normal_data.mean()
        std_val = normal_data.std()

        # Apply standardization if std > 0
        if std_val > 0:
            df_resampled[col] = (df_resampled[col] - mean_val) / std_val
        else:
            logger.warning(f"Channel {col} has zero standard deviation, skipping normalization")

    return df_resampled


def implement_temporal_split(df, validation_months=3):
    """
    Split data as per Kotowski paper

    Args:
        df (pandas.DataFrame): Telemetry data
        validation_months (int): Number of months for validation set

    Returns:
        tuple: (train_df, validation_df, test_df) dataframes
    """
    logger.info("Implementing temporal split")

    # Get time range
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()

    # Calculate midpoint (50/50 split)
    total_seconds = (end_time - start_time).total_seconds()
    mid_time = start_time + pd.Timedelta(seconds=int(total_seconds / 2))

    # Split into train and test
    train_df = df[df['timestamp'] < mid_time].copy()
    test_df = df[df['timestamp'] >= mid_time].copy()

    # Calculate validation split (last 3 months of training)
    validation_duration = pd.Timedelta(days=90)
    validation_split_time = mid_time - validation_duration

    # Split training data into train and validation
    validation_df = train_df[train_df['timestamp'] >= validation_split_time].copy()
    train_df = train_df[train_df['timestamp'] < validation_split_time].copy()

    logger.info(f"Temporal split: {len(train_df)} training, {len(validation_df)} validation, {len(test_df)} testing")

    return train_df, validation_df, test_df


def create_windows_for_group(df, group_channels, window_size=64, step_size=32):
    """
    Create feature windows for a specific channel group

    Args:
        df (pandas.DataFrame): Telemetry data
        group_channels (list): List of channel names in the group
        window_size (int): Size of sliding window
        step_size (int): Step size for sliding window

    Returns:
        tuple: (X, y, timestamps) containing feature array, label array, and timestamps
    """
    logger.info(f"Creating feature windows for channels: {group_channels} with window_size={window_size}")

    # Get feature columns for this group
    feature_cols = [col for col in group_channels if col in df.columns]

    # Skip if no features are available
    if not feature_cols:
        logger.warning(f"No feature columns found for this group")
        return None, None, None

    # Get corresponding anomaly columns
    anomaly_cols = [f'is_anomaly_{col}' for col in feature_cols if f'is_anomaly_{col}' in df.columns]

    # Skip if no anomaly columns are available
    if not anomaly_cols:
        logger.warning(f"No anomaly columns found for features {feature_cols}")
        return None, None, None

    # Initialize arrays
    X_windows = []
    y_anomaly = []
    timestamps = []

    # Create sliding windows
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]

        # Extract features
        window_features = []
        for col in feature_cols:
            # Calculate statistics for each channel
            window_features.extend([
                window[col].mean(),
                window[col].std(),
                window[col].min(),
                window[col].max()
            ])

        # Check if window contains anomalies (5% threshold as in paper)
        is_anomaly = 0
        for col in anomaly_cols:
            if window[col].mean() >= 0.05:  # 5% of the window is anomalous
                is_anomaly = 1
                break

        # Store data
        X_windows.append(window_features)
        y_anomaly.append(is_anomaly)
        timestamps.append(window['timestamp'].iloc[-1])

    # Convert to numpy arrays
    X = np.array(X_windows)
    y = np.array(y_anomaly)

    anomaly_count = sum(y)
    logger.info(f"Created {len(X)} windows with {X.shape[1]} features")
    logger.info(f"Anomaly windows: {anomaly_count} ({anomaly_count / len(y) * 100:.2f}%)")

    return X, y, timestamps