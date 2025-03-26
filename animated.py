import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as path_effects

# Set a modern style
plt.style.use('seaborn-v0_8-whitegrid')

# Define a modern color palette
COLORS = {
    'primary': '#3498db',  # Blue
    'secondary': '#2ecc71',  # Green
    'accent': '#e74c3c',  # Red
    'background': '#f8f9fa',  # Light Gray
    'text': '#2c3e50',  # Dark Blue/Gray
    'grid': '#ecf0f1',  # Light Gray for grid
    'highlight': '#f39c12'  # Orange for highlights
}

# Define font settings
FONT_FAMILY = 'Roboto, Arial, sans-serif'
TITLE_FONT = {'fontsize': 18, 'fontweight': 'bold', 'fontfamily': FONT_FAMILY, 'color': COLORS['text']}
SUBTITLE_FONT = {'fontsize': 12, 'fontstyle': 'italic', 'fontfamily': FONT_FAMILY, 'color': COLORS['text']}
AXIS_FONT = {'fontsize': 12, 'fontfamily': FONT_FAMILY, 'color': COLORS['text']}
TICK_FONT = {'fontsize': 10, 'fontfamily': FONT_FAMILY, 'color': COLORS['text']}


def setup_figure_style(fig, ax):
    """Apply consistent styling to figure and axes"""
    # Set background colors
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    # Style the grid
    ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])

    # Style the spines
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(0.8)

    # Add a subtle border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Style the ticks
    ax.tick_params(axis='both', colors=COLORS['text'], labelsize=10)

    return fig, ax


def add_data_labels(ax, x, y, labels=None, offset=(0, 5), fontsize=9, color=COLORS['text']):
    """Add data labels to points on the plot"""
    if labels is None:
        labels = [f"{val:.0f}" for val in y]

    for i, (xi, yi, label) in enumerate(zip(x, y, labels)):
        if i % 5 == 0:  # Only label every 5th point to avoid crowding
            text = ax.annotate(
                label,
                (xi, yi),
                xytext=offset,
                textcoords='offset points',
                fontsize=fontsize,
                color=color,
                ha='center'
            )
            # Add a subtle shadow effect
            text.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground='white'),
                path_effects.Normal()
            ])


def plot_channel_anomalies(metadata_file):
    """
    Reads the metadata JSON file, extracts 'num_anomalies',
    and plots a modern bar chart of anomalies for channel_X entries.
    """
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    meta = metadata[0]  # If it's a list with one dictionary

    # Extract the num_anomalies dictionary
    num_anomalies = meta["num_anomalies"]

    # Convert to DataFrame
    df_anomalies = pd.DataFrame(list(num_anomalies.items()), columns=["Label", "Num_Anomalies"])

    # Filter to keep only "channel_X" labels
    channel_pattern = re.compile(r"^channel_(\d+)$")

    def parse_channel(label):
        match = channel_pattern.match(label)
        if match:
            return int(match.group(1))
        else:
            return None

    df_anomalies["channel_num"] = df_anomalies["Label"].apply(parse_channel)
    df_anomalies = df_anomalies.dropna(subset=["channel_num"])
    df_anomalies["channel_num"] = df_anomalies["channel_num"].astype(int)

    # Sort by channel number (natural numeric order)
    df_anomalies = df_anomalies.sort_values("channel_num")

    # Find the channels with the most anomalies (for highlighting)
    highlight_threshold = df_anomalies["Num_Anomalies"].quantile(0.9)
    df_anomalies["highlight"] = df_anomalies["Num_Anomalies"] > highlight_threshold

    # Create color list with highlights
    colors = [COLORS['highlight'] if highlight else COLORS['primary']
              for highlight in df_anomalies["highlight"]]

    # Create the figure with improved styling
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    fig, ax = setup_figure_style(fig, ax)

    # Create the bar chart with a gradient effect
    bars = ax.bar(
        df_anomalies["Label"],
        df_anomalies["Num_Anomalies"],
        color=colors,
        edgecolor='white',
        linewidth=0.7,
        alpha=0.9,
        width=0.7
    )

    # Add a subtle shadow beneath each bar
    for bar in bars:
        x = bar.get_x()
        width = bar.get_width()
        height = bar.get_height()
        ax.add_patch(
            plt.Rectangle(
                (x, 0), width, height,
                fill=True,
                color='black',
                alpha=0.05,
                transform=ax.transData,
                zorder=0
            )
        )

    # Add data labels to significant bars
    for idx, row in df_anomalies[df_anomalies['highlight']].iterrows():
        ax.text(
            row['Label'],
            row['Num_Anomalies'] + 1,
            f"{row['Num_Anomalies']}",
            ha='center',
            va='bottom',
            fontsize=10,
            color=COLORS['text'],
            fontweight='bold'
        )

    # Add titles and labels with improved typography
    ax.set_title("Channel Anomaly Distribution", fontdict=TITLE_FONT, pad=20)
    ax.text(0.5, 0.97, "Number of detected anomalies per telemetry channel",
            transform=ax.transAxes, ha='center', **SUBTITLE_FONT)

    ax.set_xlabel("Channel", fontdict=AXIS_FONT, labelpad=10)
    ax.set_ylabel("Number of Anomalies", fontdict=AXIS_FONT, labelpad=10)

    # Improve x-axis
    plt.xticks(rotation=45, ha='right', **TICK_FONT)

    # Add annotations
    total_anomalies = df_anomalies["Num_Anomalies"].sum()
    max_channel = df_anomalies.loc[df_anomalies["Num_Anomalies"].idxmax(), "Label"]
    max_value = df_anomalies["Num_Anomalies"].max()

    ax.text(0.02, 0.95, f"Total Anomalies: {total_anomalies:,}",
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    ax.text(0.02, 0.90, f"Highest: {max_channel} ({max_value} anomalies)",
            transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Add a watermark/signature
    fig.text(0.95, 0.02, "Telemetry Analysis", fontsize=10, color=COLORS['text'],
             ha='right', va='bottom', alpha=0.5)

    plt.tight_layout()
    return fig


def plot_time_series(file_path, channel_name, color=COLORS['primary'], highlight_anomalies=True):
    """
    Reads a CSV file containing telemetry data and plots a modern time series.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    channel_name : str
        Name of the channel to plot (e.g., 'channel_61')
    color : str
        Color for the line plot
    highlight_anomalies : bool
        Whether to highlight potential anomalies
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])

    # Create figure with improved styling
    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    fig, ax = setup_figure_style(fig, ax)

    # Plot the main line with a modern look
    line = ax.plot(
        df["timestamp"],
        df[channel_name],
        color=color,
        linewidth=2,
        alpha=0.8,
        zorder=3
    )

    # Add a subtle shadow beneath the line
    ax.plot(
        df["timestamp"],
        df[channel_name],
        color='black',
        linewidth=4,
        alpha=0.05,
        zorder=2
    )

    # Add a subtle area fill under the line
    ax.fill_between(
        df["timestamp"],
        df[channel_name].min(),
        df[channel_name],
        color=color,
        alpha=0.1
    )

    # Highlight potential anomalies based on local statistics
    if highlight_anomalies:
        rolling_mean = df[channel_name].rolling(window=20, center=True).mean()
        rolling_std = df[channel_name].rolling(window=20, center=True).std()
        anomalies = abs(df[channel_name] - rolling_mean) > (3 * rolling_std)

        if anomalies.sum() > 0:
            ax.scatter(
                df.loc[anomalies, "timestamp"],
                df.loc[anomalies, channel_name],
                color=COLORS['accent'],
                s=80,
                alpha=0.7,
                edgecolor='white',
                linewidth=1,
                zorder=4,
                label="Potential Anomalies"
            )

    # Add a smoother trend line
    window_size = max(len(df) // 50, 5)  # Dynamic window size based on data length
    df['smooth'] = df[channel_name].rolling(window=window_size, center=True).mean()
    ax.plot(
        df["timestamp"],
        df['smooth'],
        color=COLORS['secondary'],
        linewidth=2.5,
        linestyle='-',
        alpha=0.7,
        zorder=5,
        label="Trend"
    )

    # Set title and labels with improved typography
    ax.set_title(f"Time Series Analysis: {channel_name}", fontdict=TITLE_FONT, pad=20)
    ax.text(0.5, 0.97, "Temporal patterns and anomaly detection",
            transform=ax.transAxes, ha='center', **SUBTITLE_FONT)

    ax.set_xlabel("Timestamp", fontdict=AXIS_FONT, labelpad=10)
    ax.set_ylabel(f"{channel_name} Value", fontdict=AXIS_FONT, labelpad=10)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right", **TICK_FONT)

    # Add statistics annotations
    mean_val = df[channel_name].mean()
    std_val = df[channel_name].std()

    stats_text = (
        f"Mean: {mean_val:.2f}\n"
        f"Std Dev: {std_val:.2f}\n"
        f"Range: [{df[channel_name].min():.2f}, {df[channel_name].max():.2f}]"
    )

    ax.text(0.02, 0.95, stats_text,
            transform=ax.transAxes, fontsize=11,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Add legend
    if highlight_anomalies and anomalies.sum() > 0:
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(file_path):
    """
    Reads a CSV file and plots a modern correlation heatmap among numeric channels.
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])

    # Identify only columns of the form channel_<integer>
    channel_pattern = re.compile(r"^channel_(\d+)$")
    channel_cols = [col for col in df.columns if channel_pattern.match(col)]

    def parse_channel_num(col):
        match = channel_pattern.match(col)
        if match:
            return int(match.group(1))

    channel_cols_sorted = sorted(channel_cols, key=parse_channel_num)

    # Limit to 20 channels to avoid overcrowding if there are too many
    if len(channel_cols_sorted) > 20:
        # Use a subset of channels with regular spacing
        subset_size = 20
        step = len(channel_cols_sorted) // subset_size
        channel_cols_subset = channel_cols_sorted[::step][:subset_size]
    else:
        channel_cols_subset = channel_cols_sorted

    df_channels = df[channel_cols_subset].select_dtypes(include=["number"])
    corr_matrix = df_channels.corr()

    # Clean up correlation matrix - replace NaNs with 0
    corr_matrix = corr_matrix.fillna(0)

    # Create a custom diverging colormap (professional look)
    colors = [COLORS['accent'], "#ffffff", COLORS['primary']]
    custom_cmap = LinearSegmentedColormap.from_list("modern_diverging", colors, N=256)

    # Create figure with improved styling
    fig, ax = plt.subplots(figsize=(14, 12), dpi=100)

    # Plot the heatmap with modern styling
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle

    sns.heatmap(
        corr_matrix,
        mask=mask,  # Show only lower triangle for cleaner look
        cmap=custom_cmap,
        vmin=-1,
        vmax=1,
        annot=True,  # Show correlation values
        fmt=".2f",  # Format with 2 decimal places
        square=True,  # Square cells
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        annot_kws={"size": 8},
        ax=ax
    )

    # Set title and labels with improved typography
    ax.set_title("Channel Correlation Matrix", fontdict=TITLE_FONT, pad=20)
    ax.text(0.5, 0.99, "Relationship strength between telemetry channels",
            transform=ax.transAxes, ha='center', **SUBTITLE_FONT)

    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha="right", **TICK_FONT)
    plt.yticks(**TICK_FONT)

    # Find and annotate the strongest correlations
    np.fill_diagonal(corr_matrix.values, 0)  # Exclude diagonal

    # Get strongest positive correlation
    max_corr = corr_matrix.max().max()
    max_idx = np.unravel_index(np.argmax(corr_matrix.values), corr_matrix.shape)
    pos_text = f"Strongest Positive Correlation:\n{corr_matrix.index[max_idx[0]]} & {corr_matrix.columns[max_idx[1]]}\n(r = {max_corr:.2f})"

    # Get strongest negative correlation
    min_corr = corr_matrix.min().min()
    min_idx = np.unravel_index(np.argmin(corr_matrix.values), corr_matrix.shape)
    neg_text = f"Strongest Negative Correlation:\n{corr_matrix.index[min_idx[0]]} & {corr_matrix.columns[min_idx[1]]}\n(r = {min_corr:.2f})"

    # Add annotation boxes
    fig.text(0.02, 0.02, pos_text, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    fig.text(0.78, 0.02, neg_text, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Add a count of highlighted correlations
    strong_count = (abs(corr_matrix) > 0.7).sum().sum() // 2  # Divide by 2 because matrix is symmetric
    moderate_count = ((abs(corr_matrix) > 0.4) & (abs(corr_matrix) <= 0.7)).sum().sum() // 2

    fig.text(0.5, 0.02,
             f"Strong correlations (|r| > 0.7): {strong_count}\nModerate correlations (0.4 < |r| ≤ 0.7): {moderate_count}",
             fontsize=10, ha='center',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    return fig


def plot_distribution_grid(file_path, limit=12):
    """
    Create a grid of small distribution plots for the top channels with most variation.
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])

    # Identify only columns of the form channel_<integer>
    channel_pattern = re.compile(r"^channel_(\d+)$")
    channel_cols = [col for col in df.columns if channel_pattern.match(col)]

    # Calculate coefficient of variation (std/mean) to find channels with most variation
    cv_dict = {}
    for col in channel_cols:
        if df[col].std() > 0 and abs(df[col].mean()) > 1e-10:  # Avoid div by zero
            cv_dict[col] = df[col].std() / abs(df[col].mean())

    # Get top channels by variation
    top_channels = sorted(cv_dict.items(), key=lambda x: x[1], reverse=True)[:limit]
    top_channel_names = [item[0] for item in top_channels]

    # Create a grid of small distribution plots
    n_cols = 3
    n_rows = (len(top_channel_names) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), dpi=100)
    fig.patch.set_facecolor(COLORS['background'])

    # Flatten axes array for easier indexing
    if n_rows > 1:
        axes = axes.flatten()

    for i, channel in enumerate(top_channel_names):
        if n_rows == 1:
            ax = axes[i] if n_cols > 1 else axes
        else:
            ax = axes[i]

        # Plot histogram with KDE
        sns.histplot(
            df[channel],
            kde=True,
            color=COLORS['primary'],
            alpha=0.7,
            line_kws={'linewidth': 2},
            ax=ax
        )

        # Set background
        ax.set_facecolor(COLORS['background'])

        # Style the grid
        ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])

        # Get statistics
        mean = df[channel].mean()
        median = df[channel].median()
        std = df[channel].std()

        # Add vertical lines for mean and median
        ax.axvline(mean, color=COLORS['secondary'], linestyle='-', linewidth=2, alpha=0.8, label=f'Mean: {mean:.2f}')
        ax.axvline(median, color=COLORS['accent'], linestyle='--', linewidth=2, alpha=0.8,
                   label=f'Median: {median:.2f}')

        # Add title and info
        ax.set_title(f"{channel}", fontsize=12, pad=10, fontweight='bold')
        stats_text = f"σ: {std:.2f}\nCV: {cv_dict[channel]:.2f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

        # Simplify x and y labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Add legend
        ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        if n_rows == 1:
            if n_cols > 1 and j < n_cols:
                axes[j].set_visible(False)
        else:
            axes[j].set_visible(False)

    # Add overall title
    fig.suptitle("Distribution Profiles: Channels with Highest Variability",
                 fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    return fig


if __name__ == '__main__':
    metadata_file = "data/84_months.metadata.json"
    time_series_file = "data/10_months.train.csv"

    # 1) Plot anomalies per channel from metadata
    fig1 = plot_channel_anomalies(metadata_file)
    fig1.savefig("modern_channel_anomalies.png", dpi=150, bbox_inches='tight')

    # 2.1) Plot time series for channel 61
    fig2 = plot_time_series(time_series_file, "channel_61", color=COLORS['primary'])
    fig2.savefig("modern_time_series_61.png", dpi=150, bbox_inches='tight')

    # 2.2) Plot time series for channel 40
    fig3 = plot_time_series(time_series_file, "channel_40", color=COLORS['secondary'])
    fig3.savefig("modern_time_series_40.png", dpi=150, bbox_inches='tight')

    # 3) Plot correlation heatmap
    fig4 = plot_correlation_heatmap(time_series_file)
    fig4.savefig("modern_correlation_heatmap.png", dpi=150, bbox_inches='tight')

    # 4) Add a new visualization: distribution grid
    fig5 = plot_distribution_grid(time_series_file)
    fig5.savefig("modern_distribution_grid.png", dpi=150, bbox_inches='tight')

    plt.show()