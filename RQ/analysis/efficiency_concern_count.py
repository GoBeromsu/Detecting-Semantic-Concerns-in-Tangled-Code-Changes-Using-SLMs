#!/usr/bin/env python3
"""
Efficiency Analysis: Correlation between Concern Count and Inference Time
Analyzes the relationship between concern count and inference time using Pearson correlation.
Processes raw CSV data for detailed box plot analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import argparse

# Constants
ANALYSIS_OUTPUT_DIR = Path("results/analysis")
P_VALUE_THRESHOLD = 0.05
OUTLIER_THRESHOLD_IQR = 1.5  # IQR multiplier for outlier detection

# Design constants for consistent styling
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'background': '#F5F5F5',
    'text': '#2C3E50'
}

PLOT_STYLE = {
    'figure_size': (10, 6),
    'dpi': 300,
    'line_width': 2,
    'marker_size': 60,
    'alpha': 0.7,
    'grid_alpha': 0.3
}


def setup_plot_style():
    """Setup consistent plot styling."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True
    })


def detect_outliers_iqr(data: pd.Series) -> List[int]:
    """Detect outliers using the IQR method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - OUTLIER_THRESHOLD_IQR * IQR
    upper_bound = Q3 + OUTLIER_THRESHOLD_IQR * IQR
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    return data[outlier_mask].index.tolist()


def load_csv_data(csv_paths: List[Path], remove_outliers: bool = True) -> Tuple[pd.DataFrame, dict]:
    """Load raw data from CSV files for detailed analysis.
    
    Returns:
        Tuple of (cleaned_dataframe, outlier_info)
    """
    all_data = []
    
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ['concern_count', 'inference_time']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")
        
        df['source_file'] = csv_path.name
        all_data.append(df[['concern_count', 'inference_time', 'source_file']])
    
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    # Detect outliers in inference_time
    outlier_indices = detect_outliers_iqr(combined_df['inference_time'])
    outlier_info = {
        'indices': outlier_indices,
        'count': len(outlier_indices),
        'values': combined_df.loc[outlier_indices, 'inference_time'].tolist() if outlier_indices else [],
        'removed': remove_outliers and len(outlier_indices) > 0
    }
    
    # Remove outliers if requested
    if remove_outliers and outlier_indices:
        cleaned_df = combined_df.drop(outlier_indices).reset_index(drop=True)
        print(f"Removed {len(outlier_indices)} outliers from analysis")
        print(f"Outlier values: {[f'{v:.2f}s' for v in outlier_info['values']]}")
    else:
        cleaned_df = combined_df
    
    return cleaned_df, outlier_info


def calculate_correlation(df: pd.DataFrame) -> Tuple[float, float]:
    """Calculate Pearson correlation coefficient between concern count and inference time."""
    if len(df) < 2:
        return 0.0, 1.0
        
    correlation, p_value = stats.pearsonr(df['concern_count'], df['inference_time'])
    return correlation, p_value


def perform_linear_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform linear regression analysis between concern count and inference time."""
    X = df['concern_count'].values.reshape(-1, 1)
    y = df['inference_time'].values
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate predictions and metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Calculate confidence intervals (95%)
    n = len(df)
    y_mean = np.mean(y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    
    # Standard error of regression
    mse = ss_res / (n - 2)  # degrees of freedom = n - 2 for simple linear regression
    se = np.sqrt(mse)
    
    # t-value for 95% confidence interval
    t_value = stats.t.ppf(0.975, n - 2)  # 97.5th percentile for 95% CI
    
    # Standard error for prediction
    x_mean = np.mean(X)
    ss_x = np.sum((X - x_mean) ** 2)
    
    return {
        'slope': float(model.coef_[0]),
        'intercept': float(model.intercept_),
        'r_squared': float(r2),
        'standard_error': float(se),
        't_value': float(t_value),
        'model': model,
        'predictions': y_pred,
        'x_mean': float(x_mean),
        'ss_x': float(ss_x),
        'mse': float(mse)
    }


def create_boxplot(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create box plot showing inference time distribution by concern count."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figure_size'])
    
    concern_counts = sorted(df['concern_count'].unique())
    box_data = [df[df['concern_count'] == cc]['inference_time'].values 
                for cc in concern_counts]
    
    # Create box plot with unified colors
    box_plot = ax.boxplot(box_data, labels=concern_counts, patch_artist=True,
                         boxprops=dict(facecolor=COLORS['primary'], alpha=PLOT_STYLE['alpha']),
                         medianprops=dict(color=COLORS['success'], linewidth=PLOT_STYLE['line_width']),
                         whiskerprops=dict(color=COLORS['text'], linewidth=PLOT_STYLE['line_width']),
                         capprops=dict(color=COLORS['text'], linewidth=PLOT_STYLE['line_width']),
                         flierprops=dict(marker='o', markerfacecolor=COLORS['accent'], 
                                       markeredgecolor=COLORS['accent'], markersize=4))
    
    ax.set_xlabel('Concern Count', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Inference Time (seconds)', fontweight='bold', color=COLORS['text'])
    ax.set_title('Inference Time Distribution by Concern Count', 
                fontweight='bold', color=COLORS['text'], pad=20)
    
    # Add correlation statistics with sample size
    correlation, p_value = calculate_correlation(df)
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < P_VALUE_THRESHOLD else ""
    
    stats_text = f'Pearson r = {correlation:.3f} {significance}\np = {p_value:.4f}\nn = {len(df)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['background'], 
                     alpha=0.9, edgecolor=COLORS['primary'], linewidth=1))
    
    plt.tight_layout()
    
    output_path = output_dir / "boxplot_concern_count_inference_time.png"
    plt.savefig(output_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


def create_regression_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create scatter plot with linear regression line and confidence interval."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figure_size'])
    
    # Perform linear regression
    regression_results = perform_linear_regression(df)
    
    # Create scatter plot with slight jitter to show overlapping points
    x_jitter = df['concern_count'] + np.random.normal(0, 0.03, len(df))
    
    ax.scatter(x_jitter, df['inference_time'], 
              color=COLORS['primary'], alpha=PLOT_STYLE['alpha'], 
              s=PLOT_STYLE['marker_size'], edgecolors=COLORS['text'], 
              linewidth=0.5, label='Data points')
    
    # Create regression line
    x_range = np.linspace(df['concern_count'].min(), df['concern_count'].max(), 100)
    y_pred_line = regression_results['slope'] * x_range + regression_results['intercept']
    
    ax.plot(x_range, y_pred_line, color=COLORS['secondary'], 
            linewidth=PLOT_STYLE['line_width'] + 1, alpha=0.9, 
            label=f'Regression line (R² = {regression_results["r_squared"]:.3f})')
    
    # Calculate and add 95% confidence interval
    n = len(df)
    x_mean = regression_results['x_mean']
    ss_x = regression_results['ss_x']
    mse = regression_results['mse']
    t_val = regression_results['t_value']
    
    # Standard error for each point on the line
    se_line = np.sqrt(mse * (1/n + (x_range - x_mean)**2 / ss_x))
    ci_upper = y_pred_line + t_val * se_line
    ci_lower = y_pred_line - t_val * se_line
    
    # Fill confidence interval
    ax.fill_between(x_range, ci_lower, ci_upper, 
                   color=COLORS['secondary'], alpha=0.2, 
                   label='95% Confidence interval')
    
    # Calculate and display statistics
    correlation, p_value = calculate_correlation(df)
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < P_VALUE_THRESHOLD else ""
    
    # Create regression equation text
    slope = regression_results['slope']
    intercept = regression_results['intercept']
    equation = f'y = {slope:.3f}x + {intercept:.3f}'
    
    stats_text = (f'Pearson r = {correlation:.3f} {significance}\n'
                 f'R² = {regression_results["r_squared"]:.3f}\n'
                 f'{equation}\n'
                 f'n = {len(df)}')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['background'], 
                     alpha=0.9, edgecolor=COLORS['primary'], linewidth=1))
    
    ax.set_xlabel('Concern Count', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Inference Time (seconds)', fontweight='bold', color=COLORS['text'])
    ax.set_title('Linear Regression: Concern Count vs Inference Time', 
                fontweight='bold', color=COLORS['text'], pad=20)
    
    # Set integer ticks for x-axis
    ax.set_xticks(sorted(df['concern_count'].unique()))
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    
    output_path = output_dir / "regression_concern_count_inference_time.png"
    plt.savefig(output_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path





def generate_summary_stats(df: pd.DataFrame, outlier_info: dict) -> dict:
    """Generate summary statistics for the analysis."""
    correlation, p_value = calculate_correlation(df)
    regression_results = perform_linear_regression(df)
    
    stats_dict = {
        'sample_size': len(df),
        'correlation': correlation,
        'p_value': p_value,
        'significant': p_value < P_VALUE_THRESHOLD,
        'concern_count_mean': df['concern_count'].mean(),
        'concern_count_std': df['concern_count'].std(),
        'inference_time_mean': df['inference_time'].mean(),
        'inference_time_std': df['inference_time'].std(),
        'inference_time_min': df['inference_time'].min(),
        'inference_time_max': df['inference_time'].max(),
        'outliers_detected': outlier_info['count'],
        'outliers_removed': outlier_info['removed'],
        'regression': regression_results
    }
    
    return stats_dict


def save_summary_json(stats_dict: dict, outlier_info: dict, csv_files: List[str], output_dir: Path) -> Path:
    """Save comprehensive summary as JSON."""
    from datetime import datetime, timezone
    import json
    
    # Build comprehensive summary
    summary = {
        "analysis_info": {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "analysis_type": "efficiency_concern_count_correlation",
            "input_files": csv_files,
            "outlier_detection_method": "IQR",
            "outlier_threshold": f"{OUTLIER_THRESHOLD_IQR}x IQR"
        },
        "data_summary": {
            "total_samples": stats_dict['sample_size'],
            "outliers_detected": stats_dict['outliers_detected'],
            "outliers_removed": stats_dict['outliers_removed'],
            "concern_count_range": {
                "min": 1,  # We know this from data structure
                "max": 5   # We know this from data structure  
            }
        },
        "correlation_analysis": {
            "pearson_correlation": {
                "coefficient": round(float(stats_dict['correlation']), 4),
                "p_value": float(stats_dict['p_value']),
                "significant": bool(stats_dict['significant']),
                "significance_threshold": float(P_VALUE_THRESHOLD),
                "effect_size": (
                    "large" if abs(stats_dict['correlation']) >= 0.5 else
                    "medium" if abs(stats_dict['correlation']) >= 0.3 else
                    "small" if abs(stats_dict['correlation']) >= 0.1 else
                    "negligible"
                ),
                "interpretation": (
                    "Strong positive" if stats_dict['correlation'] > 0.5 else
                    "Moderate positive" if stats_dict['correlation'] > 0.3 else
                    "Weak positive" if stats_dict['correlation'] > 0.1 else
                    "Weak negative" if stats_dict['correlation'] > -0.1 else
                    "Moderate negative" if stats_dict['correlation'] > -0.3 else
                    "Strong negative"
                )
            }
        },
        "linear_regression": {
            "equation": {
                "slope": round(float(stats_dict['regression']['slope']), 4),
                "intercept": round(float(stats_dict['regression']['intercept']), 4),
                "formula": f"y = {stats_dict['regression']['slope']:.4f}x + {stats_dict['regression']['intercept']:.4f}"
            },
            "model_fit": {
                "r_squared": round(float(stats_dict['regression']['r_squared']), 4),
                "standard_error": round(float(stats_dict['regression']['standard_error']), 4),
                "interpretation": (
                    "Excellent fit" if stats_dict['regression']['r_squared'] >= 0.9 else
                    "Good fit" if stats_dict['regression']['r_squared'] >= 0.7 else
                    "Moderate fit" if stats_dict['regression']['r_squared'] >= 0.5 else
                    "Weak fit" if stats_dict['regression']['r_squared'] >= 0.3 else
                    "Poor fit"
                )
            },
            "confidence_interval": {
                "level": 95,
                "t_value": round(float(stats_dict['regression']['t_value']), 4)
            }
        },
        "descriptive_statistics": {
            "concern_count": {
                "mean": round(float(stats_dict['concern_count_mean']), 2),
                "std": round(float(stats_dict['concern_count_std']), 2)
            },
            "inference_time": {
                "mean": round(float(stats_dict['inference_time_mean']), 4),
                "std": round(float(stats_dict['inference_time_std']), 4),
                "min": round(float(stats_dict['inference_time_min']), 4),
                "max": round(float(stats_dict['inference_time_max']), 4),
                "unit": "seconds"
            }
        },
        "outlier_analysis": {
            "detection_method": "Interquartile Range (IQR)",
            "threshold": f"Q1 - {OUTLIER_THRESHOLD_IQR}×IQR and Q3 + {OUTLIER_THRESHOLD_IQR}×IQR",
            "total_detected": outlier_info['count'],
            "outlier_values": [round(v, 2) for v in outlier_info['values']] if outlier_info['values'] else [],
            "impact": {
                "removed_from_analysis": bool(outlier_info['removed']),
                "percentage_of_data": round((outlier_info['count'] / (stats_dict['sample_size'] + outlier_info['count'])) * 100, 1) if outlier_info['count'] > 0 else 0
            }
        }
    }
    
    output_path = output_dir / "efficiency_analysis_summary.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Analyze concern count vs inference time correlation from CSV files')
    parser.add_argument('csv_files', nargs='+', help='CSV files to analyze')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--keep-outliers', action='store_true', 
                       help='Keep outliers in analysis (default: remove outliers)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # Validate CSV structure consistency across files
        expected_columns = {'concern_count', 'inference_time'}
        csv_paths = [Path(f) for f in args.csv_files]
        
        for csv_path in csv_paths:
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            df_sample = pd.read_csv(csv_path, nrows=1)
            available_columns = set(df_sample.columns)
            
            if not expected_columns.issubset(available_columns):
                missing = expected_columns - available_columns
                raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")
        
        # Generate descriptive output directory name  
        file_stems = [Path(f).stem for f in args.csv_files]
        files_summary = "_".join(file_stems)[:50]
        outlier_suffix = "_with_outliers" if args.keep_outliers else "_clean"
        output_dir = ANALYSIS_OUTPUT_DIR / f"ef_{files_summary}{outlier_suffix}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting efficiency analysis: Concern Count vs Inference Time")
    print("=" * 60)
    
    # Process CSV files
    print(f"Processing {len(args.csv_files)} CSV file(s)...")
    csv_paths = [Path(f) for f in args.csv_files]
    
    try:
        # Load data with outlier handling
        remove_outliers = not args.keep_outliers
        df, outlier_info = load_csv_data(csv_paths, remove_outliers=remove_outliers)
        print(f"Loaded {len(df)} data points from CSV files")
        
        if outlier_info['count'] > 0:
            print(f"Detected {outlier_info['count']} outliers in inference time")
            print(f"Outlier detection method: IQR (Interquartile Range)")
            print(f"Outlier threshold: Q1 - {OUTLIER_THRESHOLD_IQR}×IQR and Q3 + {OUTLIER_THRESHOLD_IQR}×IQR")
            if outlier_info['removed']:
                print(f"Outliers removed from analysis")
            else:
                print(f"Outliers kept in analysis (use --keep-outliers flag)")
        else:
            print("No outliers detected in inference time")
        
        # Generate statistics
        stats = generate_summary_stats(df, outlier_info)
        
        # Print basic results
        print(f"\nCorrelation Analysis Results:")
        print(f"- Sample size: {stats['sample_size']}")
        print(f"- Outliers detected: {stats['outliers_detected']}")
        print(f"- Outliers removed: {'Yes' if stats['outliers_removed'] else 'No'}")
        print(f"- Pearson correlation: r = {stats['correlation']:.4f}")
        print(f"- P-value: {stats['p_value']:.6f}")
        print(f"- Significant: {'Yes' if stats['significant'] else 'No'}")
        
        # Generate individual visualizations
        print(f"\nGenerating visualizations...")
        boxplot_path = create_boxplot(df, output_dir)
        regression_path = create_regression_plot(df, output_dir)
        
        # Save results
        json_path = save_summary_json(stats, outlier_info, args.csv_files, output_dir)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"- Box plot: {boxplot_path.name}")
        print(f"- Regression plot: {regression_path.name}")
        print(f"- Summary JSON: {json_path.name}")
        print(f"\nRegression equation: y = {stats['regression']['slope']:.4f}x + {stats['regression']['intercept']:.4f}")
        print(f"R² = {stats['regression']['r_squared']:.4f}")
        
    except Exception as e:
        print(f"Error processing CSV files: {e}")
        raise


if __name__ == "__main__":
    main()
