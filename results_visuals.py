import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Use real group results data
group_results = {
  "3": {
    "results": {
      "precision": 0.0,
      "recall": 0.0,
      "f_beta": 0.0,
      "event_precision": 0.0,
      "event_recall": 0.0,
      "event_f_beta": 0.0,
      "alarming_precision": 1.0,
      "adtqc_score": 0.0,
      "adtqc_after_ratio": 0.0,
      "confusion_matrix": [[6924, 263], [0, 0]]
    },
    "channels": ["channel_12", "channel_13", "channel_19", "channel_20", "channel_27", "channel_28", "channel_36", "channel_37"]
  },
  "4": {
    "results": {
      "precision": 0.02287581699346405,
      "recall": 0.2,
      "f_beta": 0.02779984114376489,
      "event_precision": 0.10177509250152217,
      "event_recall": 0.4,
      "event_f_beta": 0.11961050899142965,
      "alarming_precision": 1.0,
      "adtqc_score": 0.0,
      "adtqc_after_ratio": 0.0,
      "confusion_matrix": [[6519, 598], [56, 14]]
    },
    "channels": ["channel_14", "channel_21", "channel_29", "channel_30", "channel_38"]
  },
  "8": {
    "results": {
      "precision": 0.11173184357541899,
      "recall": 0.7843137254901961,
      "f_beta": 0.1348617666891436,
      "event_precision": 0.19108744394618837,
      "event_recall": 1.0,
      "event_f_beta": 0.22796881060332494,
      "alarming_precision": 1.0,
      "adtqc_score": 0.0,
      "adtqc_after_ratio": 0.0,
      "confusion_matrix": [[6818, 318], [11, 40]]
    },
    "channels": ["channel_41", "channel_42", "channel_43", "channel_44", "channel_45", "channel_46"]
  },
  "13": {
    "results": {
      "precision": 0.2041522491349481,
      "recall": 0.6555555555555556,
      "f_beta": 0.23675762439807385,
      "event_precision": 0.0604744962660279,
      "event_recall": 1.0,
      "event_f_beta": 0.07446727755788105,
      "alarming_precision": 1.0,
      "adtqc_score": 0.0,
      "adtqc_after_ratio": 0.0,
      "confusion_matrix": [[6867, 230], [31, 59]]
    },
    "channels": ["channel_57", "channel_58", "channel_59", "channel_60"]
  },
  "18": {
    "results": {
      "precision": 0.11812627291242363,
      "recall": 0.9354838709677419,
      "f_beta": 0.14313919052319843,
      "event_precision": 0.3354385964912281,
      "event_recall": 0.8333333333333334,
      "event_f_beta": 0.3809614894159653,
      "alarming_precision": 0.8,
      "adtqc_score": 1.0,
      "adtqc_after_ratio": 1.0,
      "confusion_matrix": [[6692, 433], [4, 58]]
    },
    "channels": ["channel_70", "channel_71", "channel_72", "channel_73", "channel_74", "channel_75", "channel_76"]
  }
}

# Extract group numbers and their corresponding metrics
groups = []
event_f_betas = []
precisions = []
recalls = []
channel_counts = []
alarming_precs = []

for group_num, data in group_results.items():
    groups.append(f"Group {group_num}")
    event_f_betas.append(data['results']['event_f_beta'])
    precisions.append(data['results']['precision'])
    recalls.append(data['results']['recall'])
    channel_counts.append(len(data['channels']))
    alarming_precs.append(data['results']['alarming_precision'])

# Create a DataFrame for visualization
df = pd.DataFrame({
    'Group': groups,
    'Event F0.5': event_f_betas,
    'Precision': precisions,
    'Recall': recalls,
    'Channel Count': channel_counts,
    'Alarming Precision': alarming_precs
})

# Create the bubble chart
plt.figure(figsize=(12, 8))

# Scatter plot with bubble size representing number of channels
scatter = plt.scatter(df['Precision'], df['Recall'],
            s=df['Channel Count']*100,  # Scale bubble size
            c=df['Event F0.5'],  # Color by Event F0.5 score
            cmap='viridis',
            alpha=0.7)

# Add group labels next to each bubble
for i, group in enumerate(df['Group']):
    plt.annotate(group,
                 (df['Precision'][i], df['Recall'][i]),
                 xytext=(7, 0),
                 textcoords='offset points',
                 fontsize=10,
                 fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Event F0.5 Score', fontsize=12)

# Set labels and title
plt.xlabel('Precision', fontsize=14)
plt.ylabel('Recall', fontsize=14)
plt.title('Channel Group Performance Comparison', fontsize=16)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Explanation text in corner
textstr = "Bubble size represents\nnumber of channels in group"
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.figtext(0.15, 0.02, textstr, fontsize=12, bbox=props)

plt.tight_layout()

plt.show()