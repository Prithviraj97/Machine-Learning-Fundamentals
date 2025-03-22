import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Sample data structure (you'll replace with your actual measurements)
metrics = {
    'Platform': ['GPU', 'Arduino Nano BLE 33'],
    'Avg Inference Time (ms)': [15, 250],  # Example values
    'Memory Usage (MB)': [120, 0.136],  # Example values
    'Power Consumption (W)': [150, 0.015],  # Example values
    'Accuracy (%)': [92, 85]  # Example values
}

df = pd.DataFrame(metrics)

# Set style for plots
sns.set(style="whitegrid")
plt.figure(figsize=(20, 15))

# 1. Inference Time Comparison
plt.subplot(2, 2, 1)
sns.barplot(x='Platform', y='Inference Time (ms)', data=df, palette='viridis')
plt.title('Inference Time Comparison', fontsize=16)
plt.ylabel('Time (ms)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 2. Memory Usage Comparison (log scale)
plt.subplot(2, 2, 2)
memory_plot = sns.barplot(x='Platform', y='Memory Usage (MB)', data=df, palette='viridis')
plt.title('Memory Usage Comparison', fontsize=16)
plt.ylabel('Memory (MB)', fontsize=14)
plt.yscale('log')  # Log scale for better comparison
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add value labels on bars
for i, value in enumerate(df['Memory Usage (MB)']):
    memory_plot.text(i, value/10, f'{value} MB', ha='center', fontsize=12)

# 3. Power Consumption Comparison (log scale)
plt.subplot(2, 2, 3)
power_plot = sns.barplot(x='Platform', y='Power Consumption (W)', data=df, palette='viridis')
plt.title('Power Consumption Comparison', fontsize=16)
plt.ylabel('Power (W)', fontsize=14)
plt.yscale('log')  # Log scale for better comparison
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add value labels on bars
for i, value in enumerate(df['Power Consumption (W)']):
    power_plot.text(i, value/10, f'{value} W', ha='center', fontsize=12)

# 4. Accuracy Comparison
plt.subplot(2, 2, 4)
accuracy_plot = sns.barplot(x='Platform', y='Accuracy (%)', data=df, palette='viridis')
plt.title('Model Accuracy Comparison', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.ylim(0, 100)  # Set y-axis from 0 to 100
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add value labels on bars
for i, value in enumerate(df['Accuracy (%)']):
    accuracy_plot.text(i, value-5, f'{value}%', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Radar Chart for Overall Comparison
# Normalize metrics for radar chart
normalized_metrics = df.copy()
for column in df.columns[1:]:  # Skip the 'Platform' column
    if column == 'Inference Time (ms)' or column == 'Memory Usage (MB)' or column == 'Power Consumption (W)':
        # For these metrics, lower is better, so invert the normalization
        max_val = df[column].max()
        normalized_metrics[column] = 1 - (df[column] / max_val)
    else:
        # For accuracy, higher is better
        min_val = df[column].min()
        max_val = df[column].max()
        normalized_metrics[column] = (df[column] - min_val) / (max_val - min_val)

# Set up the radar chart
categories = df.columns[1:]  # Skip the 'Platform' column
N = len(categories)

# Create the angle for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Create the plot
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# Draw one platform at a time
for i, platform in enumerate(df['Platform']):
    values = normalized_metrics.iloc[i, 1:].values.tolist()
    values += values[:1]  # Close the loop
    
    # Plot the values
    ax.plot(angles, values, linewidth=2, label=platform)
    ax.fill(angles, values, alpha=0.25)

# Set the category labels
plt.xticks(angles[:-1], categories, fontsize=12)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)

plt.title('Performance Metrics Comparison', fontsize=16)
plt.tight_layout()
plt.savefig('radar_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Create a workflow diagram for the poster
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.figure(figsize=(12, 8))
ax = plt.gca()

# Define the boxes
boxes = {
    'train': Rectangle((0.1, 0.7), 0.2, 0.2, fill=True, color='#4CAF50', alpha=0.7),
    'eval_gpu': Rectangle((0.4, 0.7), 0.2, 0.2, fill=True, color='#2196F3', alpha=0.7),
    'convert': Rectangle((0.7, 0.7), 0.2, 0.2, fill=True, color='#FFC107', alpha=0.7),
    'deploy': Rectangle((0.4, 0.4), 0.2, 0.2, fill=True, color='#FF5722', alpha=0.7),
    'compare': Rectangle((0.4, 0.1), 0.2, 0.2, fill=True, color='#9C27B0', alpha=0.7)
}

# Add boxes to plot
for box in boxes.values():
    ax.add_patch(box)

# Add arrows
arrows = [
    FancyArrowPatch((0.3, 0.8), (0.4, 0.8), arrowstyle='->', mutation_scale=20, color='black'),
    FancyArrowPatch((0.6, 0.8), (0.7, 0.8), arrowstyle='->', mutation_scale=20, color='black'),
    FancyArrowPatch((0.8, 0.7), (0.5, 0.6), arrowstyle='->', mutation_scale=20, color='black'),
    FancyArrowPatch((0.5, 0.4), (0.5, 0.3), arrowstyle='->', mutation_scale=20, color='black')
]

for arrow in arrows:
    ax.add_patch(arrow)

# Add text
plt.text(0.2, 0.8, 'Train Model\non GPU', ha='center', va='center', fontsize=12)
plt.text(0.5, 0.8, 'Evaluate on\nGPU', ha='center', va='center', fontsize=12)
plt.text(0.8, 0.8, 'Convert to\nTFLite', ha='center', va='center', fontsize=12)
plt.text(0.5, 0.5, 'Deploy on\nArduino', ha='center', va='center', fontsize=12)
plt.text(0.5, 0.2, 'Compare\nMetrics', ha='center', va='center', fontsize=12)

# Remove axis
plt.axis('off')
plt.tight_layout()
plt.savefig('workflow_diagram.png', dpi=300, bbox_inches='tight')
plt.show()
