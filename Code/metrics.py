import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_metrics_data():
    """Load the metrics data from CSV file"""
    # Get the absolute path to the project root
    project_root = Path(__file__).parent.parent
    data_path = project_root / "Data" / "metrics.csv"
    df = pd.read_csv(data_path)
    
    # Convert numeric columns, handling empty strings as NaN
    numeric_columns = ['Precision', 'Recall', 'Accuracy', 'F1', 'ROC-AUC', 'PR-AUC']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def create_line_plots(df):
    """Create line plots showing metric progression across stages for each model"""
    metrics = ['Precision', 'Recall', 'Accuracy', 'F1', 'ROC-AUC', 'PR-AUC']
    models = df['Model'].unique()
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for model in models:
            model_data = df[df['Model'] == model]
            # Only plot if we have data for this metric
            if not model_data[metric].isna().all():
                ax.plot(model_data['Stage'], model_data[metric], 
                       marker='o', linewidth=2, markersize=8, label=model)
        
        ax.set_title(f'{metric} Across Stages', fontsize=14, fontweight='bold')
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    # Get the absolute path to the Images directory
    project_root = Path(__file__).parent.parent
    images_path = project_root / "Images" / "metrics_line_plots.png"
    plt.savefig(images_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_heatmap(df):
    """Create heatmap showing all metrics for all models and stages"""
    # Pivot the data for heatmap
    metrics = ['Precision', 'Recall', 'Accuracy', 'F1', 'ROC-AUC', 'PR-AUC']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create pivot table for this metric
        pivot_data = df.pivot(index='Model', columns='Stage', values=metric)
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='viridis', 
                   ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(f'{metric} Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    # Get the absolute path to the Images directory
    project_root = Path(__file__).parent.parent
    images_path = project_root / "Images" / "metrics_heatmaps.png"
    plt.savefig(images_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_radar_chart(df):
    """Create radar charts for each model showing all metrics"""
    metrics = ['Precision', 'Recall', 'Accuracy', 'F1', 'ROC-AUC', 'PR-AUC']
    models = df['Model'].unique()
    stages = df['Stage'].unique()
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, axes = plt.subplots(1, len(stages), figsize=(20, 6))
    if len(stages) == 1:
        axes = [axes]
    
    for stage_idx, stage in enumerate(stages):
        ax = axes[stage_idx]
        
        for model in models:
            model_data = df[(df['Model'] == model) & (df['Stage'] == stage)]
            if not model_data.empty:
                values = []
                for metric in metrics:
                    val = model_data[metric].iloc[0]
                    values.append(val if not pd.isna(val) else 0)
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model)
                ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(f'{stage} Stage', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    # Get the absolute path to the Images directory
    project_root = Path(__file__).parent.parent
    images_path = project_root / "Images" / "metrics_radar_charts.png"
    plt.savefig(images_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_bar_comparison(df):
    """Create grouped bar charts comparing metrics across stages"""
    metrics = ['Precision', 'Recall', 'Accuracy', 'F1', 'ROC-AUC', 'PR-AUC']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create grouped bar chart
        pivot_data = df.pivot(index='Model', columns='Stage', values=metric)
        
        pivot_data.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title(f'{metric} Comparison Across Stages', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.legend(title='Stage')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Get the absolute path to the Images directory
    project_root = Path(__file__).parent.parent
    images_path = project_root / "Images" / "metrics_bar_comparison.png"
    plt.savefig(images_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_improvement_analysis(df):
    """Create analysis showing improvement from Original to Optimized stage"""
    metrics = ['Precision', 'Recall', 'Accuracy', 'F1', 'ROC-AUC', 'PR-AUC']
    
    # Calculate improvement (Optimized - Original)
    improvement_data = []
    
    for model in df['Model'].unique():
        original = df[(df['Model'] == model) & (df['Stage'] == 'Original')]
        optimized = df[(df['Model'] == model) & (df['Stage'] == 'Optimized')]
        
        if not original.empty and not optimized.empty:
            improvements = {}
            for metric in metrics:
                orig_val = original[metric].iloc[0] if not pd.isna(original[metric].iloc[0]) else 0
                opt_val = optimized[metric].iloc[0] if not pd.isna(optimized[metric].iloc[0]) else 0
                improvements[metric] = opt_val - orig_val
            
            improvements['Model'] = model
            improvement_data.append(improvements)
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Create improvement visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model in enumerate(improvement_df['Model']):
        model_data = improvement_df[improvement_df['Model'] == model]
        values = [model_data[metric].iloc[0] for metric in metrics]
        
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Improvement (Optimized - Original)', fontsize=12)
    ax.set_title('Model Performance Improvement from Original to Optimized', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    # Get the absolute path to the Images directory
    project_root = Path(__file__).parent.parent
    images_path = project_root / "Images" / "metrics_improvement_analysis.png"
    plt.savefig(images_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return improvement_df

def print_summary_statistics(df):
    """Print summary statistics for the metrics"""
    print("=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    
    metrics = ['Precision', 'Recall', 'Accuracy', 'F1', 'ROC-AUC', 'PR-AUC']
    
    for stage in df['Stage'].unique():
        print(f"\n{stage.upper()} STAGE:")
        print("-" * 30)
        stage_data = df[df['Stage'] == stage]
        
        for metric in metrics:
            print(f"\n{metric}:")
            for model in stage_data['Model'].unique():
                model_data = stage_data[stage_data['Model'] == model]
                value = model_data[metric].iloc[0]
                if not pd.isna(value):
                    print(f"  {model}: {value:.4f}")
                else:
                    print(f"  {model}: N/A")

def main():
    """Main function to run all visualizations"""
    print("Loading metrics data...")
    df = load_metrics_data()
    
    print("Creating line plots...")
    create_line_plots(df)
    
    print("Creating heatmaps...")
    create_heatmap(df)
    
    print("Creating radar charts...")
    create_radar_chart(df)
    
    print("Creating bar comparison charts...")
    create_bar_comparison(df)
    
    print("Creating improvement analysis...")
    improvement_df = create_improvement_analysis(df)
    
    print("Printing summary statistics...")
    print_summary_statistics(df)
    
    print("\n" + "=" * 60)
    print("All visualizations have been saved to the Images/ folder:")
    print("- metrics_line_plots.png")
    print("- metrics_heatmaps.png") 
    print("- metrics_radar_charts.png")
    print("- metrics_bar_comparison.png")
    print("- metrics_improvement_analysis.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
