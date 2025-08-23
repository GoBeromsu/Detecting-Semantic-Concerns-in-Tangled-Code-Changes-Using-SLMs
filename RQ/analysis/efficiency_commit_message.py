#!/usr/bin/env python3
"""
Efficiency Analysis: Correlation between Commit Message and Inference Time
Analyzes the relationship between commit message presence and inference time using statistical tests.
Processes raw CSV data for detailed box plot analysis and group comparison.
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

# Constants - Use root results directory (from project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up from RQ/analysis/ to project root
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"
P_VALUE_THRESHOLD = 0.05
OUTLIER_THRESHOLD_IQR = 1.5  # IQR multiplier for outlier detection

# Design constants for consistent styling
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'background': '#F5F5F5',
    'text': '#2C3E50',
    'with_message': '#2E86AB',
    'without_message': '#A23B72'
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
        required_cols = ['with_message', 'inference_time']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")
        
        df['source_file'] = csv_path.name
        all_data.append(df[['with_message', 'inference_time', 'source_file']])
    
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


def calculate_group_comparison(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate statistical comparison between with/without message groups."""
    with_msg = df[df['with_message'] == True]['inference_time']
    without_msg = df[df['with_message'] == False]['inference_time']
    
    # Basic statistics
    stats_dict = {
        'with_message': {
            'count': len(with_msg),
            'mean': float(with_msg.mean()),
            'std': float(with_msg.std()),
            'median': float(with_msg.median()),
            'min': float(with_msg.min()),
            'max': float(with_msg.max())
        },
        'without_message': {
            'count': len(without_msg),
            'mean': float(without_msg.mean()),
            'std': float(without_msg.std()),
            'median': float(without_msg.median()),
            'min': float(without_msg.min()),
            'max': float(without_msg.max())
        }
    }
    
    # Statistical tests
    if len(with_msg) > 1 and len(without_msg) > 1:
        # Independent t-test
        t_stat, t_pvalue = stats.ttest_ind(with_msg, without_msg)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(with_msg, without_msg, alternative='two-sided')
        
        # Cohen's d effect size
        pooled_std = np.sqrt(((len(with_msg) - 1) * with_msg.var() + 
                             (len(without_msg) - 1) * without_msg.var()) / 
                            (len(with_msg) + len(without_msg) - 2))
        cohens_d = (with_msg.mean() - without_msg.mean()) / pooled_std
        
        stats_dict['statistical_tests'] = {
            't_test': {
                'statistic': float(t_stat),
                'p_value': float(t_pvalue),
                'significant': bool(t_pvalue < P_VALUE_THRESHOLD)
            },
            'mann_whitney_u': {
                'statistic': float(u_stat),
                'p_value': float(u_pvalue),
                'significant': bool(u_pvalue < P_VALUE_THRESHOLD)
            },
            'effect_size': {
                'cohens_d': float(cohens_d),
                'interpretation': (
                    'Large' if abs(cohens_d) >= 0.8 else
                    'Medium' if abs(cohens_d) >= 0.5 else
                    'Small' if abs(cohens_d) >= 0.2 else
                    'Negligible'
                )
            }
        }
    else:
        stats_dict['statistical_tests'] = None
    
    return stats_dict


def calculate_point_biserial_correlation(df: pd.DataFrame) -> Tuple[float, float]:
    """Calculate point-biserial correlation between binary variable and continuous variable."""
    # Convert boolean to numeric (True=1, False=0)
    with_message_numeric = df['with_message'].astype(int)
    correlation, p_value = stats.pearsonr(with_message_numeric, df['inference_time'])
    return correlation, p_value


def perform_linear_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform linear regression analysis between message presence and inference time."""
    X = df['with_message'].astype(int).values.reshape(-1, 1)  # Convert boolean to int
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
    
    # Standard error of regression
    mse = ss_res / (n - 2)  # degrees of freedom = n - 2 for simple linear regression
    se = np.sqrt(mse)
    
    # t-value for 95% confidence interval
    t_value = stats.t.ppf(0.975, n - 2)  # 97.5th percentile for 95% CI
    
    return {
        'slope': float(model.coef_[0]),
        'intercept': float(model.intercept_),
        'r_squared': float(r2),
        'standard_error': float(se),
        't_value': float(t_value),
        'model': model,
        'predictions': y_pred
    }


def create_boxplot(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create box plot showing inference time distribution by commit message presence."""
    setup_plot_style()
    
    # Use consistent figure size
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figure_size'])
    
    # Prepare data for box plot
    with_msg_data = df[df['with_message'] == True]['inference_time']
    without_msg_data = df[df['with_message'] == False]['inference_time']
    
    box_data = [without_msg_data, with_msg_data]
    labels = ['Without Message', 'With Message']
    colors = [COLORS['without_message'], COLORS['with_message']]
    
    # Create box plot with adjusted width and positions
    positions = [1, 2]  # Closer positions
    box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True,
                         widths=0.6,  # Make boxes slightly wider
                         boxprops=dict(alpha=PLOT_STYLE['alpha']),
                         medianprops=dict(color=COLORS['success'], linewidth=PLOT_STYLE['line_width']),
                         whiskerprops=dict(color=COLORS['text'], linewidth=PLOT_STYLE['line_width']),
                         capprops=dict(color=COLORS['text'], linewidth=PLOT_STYLE['line_width']),
                         flierprops=dict(marker='o', markerfacecolor=COLORS['accent'], 
                                       markeredgecolor=COLORS['accent'], markersize=4))
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Set custom x-axis labels and ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(0.5, 2.5)  # Tighter x-axis limits
    
    ax.set_xlabel('Commit Message Presence', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Inference Time (seconds)', fontweight='bold', color=COLORS['text'])
    ax.set_title('Inference Time Distribution by Commit Message Presence', 
                fontweight='bold', color=COLORS['text'], pad=20)
    
    # Add statistical information
    group_stats = calculate_group_comparison(df)
    correlation, p_value = calculate_point_biserial_correlation(df)
    
    # Perform statistical test
    if group_stats['statistical_tests']:
        t_test = group_stats['statistical_tests']['t_test']
        mann_whitney = group_stats['statistical_tests']['mann_whitney_u']
        effect_size = group_stats['statistical_tests']['effect_size']
        
        significance = "***" if t_test['p_value'] < 0.001 else "**" if t_test['p_value'] < 0.01 else "*" if t_test['p_value'] < P_VALUE_THRESHOLD else ""
        
        stats_text = (f'Point-biserial r = {correlation:.3f}\n'
                      f't-test p = {t_test["p_value"]:.4f} {significance}\n'
                      f'Cohen\'s d = {effect_size["cohens_d"]:.3f}\n'
                      f'n = {len(df)}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['background'], 
                         alpha=0.9, edgecolor=COLORS['primary'], linewidth=1))
    
    # Add legend for box plot components
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLORS['without_message'], alpha=PLOT_STYLE['alpha'], label='Without Message (IQR)'),
        Patch(facecolor=COLORS['with_message'], alpha=PLOT_STYLE['alpha'], label='With Message (IQR)'),
        Line2D([0], [0], color=COLORS['success'], linewidth=PLOT_STYLE['line_width'], label='Median'),
        Line2D([0], [0], color=COLORS['text'], linewidth=PLOT_STYLE['line_width'], label='Whiskers (1.5×IQR)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['accent'], 
               markersize=4, label='Outliers')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    
    output_path = output_dir / "boxplot_commit_message_inference_time.png"
    plt.savefig(output_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


def create_regression_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create scatter plot with linear regression line for message presence vs inference time."""
    setup_plot_style()
    
    # Use consistent figure size
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figure_size'])
    
    # Perform linear regression
    regression_results = perform_linear_regression(df)
    
    # Prepare data with jitter for visualization
    x_vals = df['with_message'].astype(int)
    x_jitter = x_vals + np.random.normal(0, 0.03, len(x_vals))
    
    # Create scatter plot
    colors = [COLORS['without_message'] if not msg else COLORS['with_message'] 
              for msg in df['with_message']]
    
    ax.scatter(x_jitter, df['inference_time'], 
              c=colors, alpha=PLOT_STYLE['alpha'], 
              s=PLOT_STYLE['marker_size'], edgecolors=COLORS['text'], 
              linewidth=0.5)
    
    # Create regression line
    x_range = np.array([0, 1])
    y_pred_line = regression_results['slope'] * x_range + regression_results['intercept']
    
    ax.plot(x_range, y_pred_line, color=COLORS['secondary'], 
            linewidth=PLOT_STYLE['line_width'] + 1, alpha=0.9, 
            label=f'Regression line (R² = {regression_results["r_squared"]:.3f})')
    
    # Calculate and display statistics
    correlation, p_value = calculate_point_biserial_correlation(df)
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < P_VALUE_THRESHOLD else ""
    
    # Create regression equation text
    slope = regression_results['slope']
    intercept = regression_results['intercept']
    equation = f'y = {slope:.3f}x + {intercept:.3f}'
    
    stats_text = (f'Point-biserial r = {correlation:.3f} {significance}\n'
                 f'R² = {regression_results["r_squared"]:.3f}\n'
                 f'{equation}\n'
                 f'n = {len(df)}')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['background'], 
                     alpha=0.9, edgecolor=COLORS['primary'], linewidth=1))
    
    ax.set_xlabel('Commit Message Presence (0=No, 1=Yes)', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Inference Time (seconds)', fontweight='bold', color=COLORS['text'])
    ax.set_title('Linear Regression: Commit Message vs Inference Time', 
                fontweight='bold', color=COLORS['text'], pad=20)
    
    # Set x-axis ticks and limits for tighter spacing
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Without Message', 'With Message'])
    ax.set_xlim(-0.2, 1.2)  # Tighter x-axis limits
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    output_path = output_dir / "regression_commit_message_inference_time.png"
    plt.savefig(output_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


def generate_summary_stats(df: pd.DataFrame, outlier_info: dict) -> dict:
    """Generate summary statistics for the analysis."""
    group_stats = calculate_group_comparison(df)
    correlation, p_value = calculate_point_biserial_correlation(df)
    regression_results = perform_linear_regression(df)
    
    stats_dict = {
        'sample_size': len(df),
        'correlation': correlation,
        'p_value': p_value,
        'significant': p_value < P_VALUE_THRESHOLD,
        'group_statistics': group_stats,
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
            "analysis_type": "efficiency_commit_message_correlation",
            "input_files": csv_files,
            "outlier_detection_method": "IQR",
            "outlier_threshold": f"{OUTLIER_THRESHOLD_IQR}x IQR"
        },
        "data_summary": {
            "total_samples": stats_dict['sample_size'],
            "outliers_detected": stats_dict['outliers_detected'],
            "outliers_removed": stats_dict['outliers_removed'],
            "with_message_count": stats_dict['group_statistics']['with_message']['count'],
            "without_message_count": stats_dict['group_statistics']['without_message']['count']
        },
        "correlation_analysis": {
            "point_biserial_correlation": {
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
        "group_comparison": {
            "with_message": {
                "count": stats_dict['group_statistics']['with_message']['count'],
                "mean": round(stats_dict['group_statistics']['with_message']['mean'], 4),
                "std": round(stats_dict['group_statistics']['with_message']['std'], 4),
                "median": round(stats_dict['group_statistics']['with_message']['median'], 4),
                "min": round(stats_dict['group_statistics']['with_message']['min'], 4),
                "max": round(stats_dict['group_statistics']['with_message']['max'], 4)
            },
            "without_message": {
                "count": stats_dict['group_statistics']['without_message']['count'],
                "mean": round(stats_dict['group_statistics']['without_message']['mean'], 4),
                "std": round(stats_dict['group_statistics']['without_message']['std'], 4),
                "median": round(stats_dict['group_statistics']['without_message']['median'], 4),
                "min": round(stats_dict['group_statistics']['without_message']['min'], 4),
                "max": round(stats_dict['group_statistics']['without_message']['max'], 4)
            }
        },
        "statistical_tests": stats_dict['group_statistics']['statistical_tests'],
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
    
    output_path = output_dir / "efficiency_commit_message_summary.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Analyze commit message vs inference time correlation from CSV files')
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
        expected_columns = {'with_message', 'inference_time'}
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
        output_dir = ANALYSIS_OUTPUT_DIR / f"cm_{files_summary}{outlier_suffix}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting efficiency analysis: Commit Message vs Inference Time")
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
        print(f"\nGroup Comparison Results:")
        print(f"- Sample size: {stats['sample_size']}")
        print(f"- With message: {stats['group_statistics']['with_message']['count']} samples")
        print(f"- Without message: {stats['group_statistics']['without_message']['count']} samples")
        print(f"- Point-biserial correlation: r = {stats['correlation']:.4f}")
        print(f"- P-value: {stats['p_value']:.6f}")
        print(f"- Significant: {'Yes' if stats['significant'] else 'No'}")
        
        if stats['group_statistics']['statistical_tests']:
            t_test = stats['group_statistics']['statistical_tests']['t_test']
            print(f"- t-test p-value: {t_test['p_value']:.6f}")
            print(f"- t-test significant: {'Yes' if t_test['significant'] else 'No'}")
            
            effect_size = stats['group_statistics']['statistical_tests']['effect_size']
            print(f"- Cohen's d: {effect_size['cohens_d']:.4f} ({effect_size['interpretation']})")
        
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
        
        # Print group means for interpretation
        with_msg_mean = stats['group_statistics']['with_message']['mean']
        without_msg_mean = stats['group_statistics']['without_message']['mean']
        print(f"\nGroup means:")
        print(f"- With message: {with_msg_mean:.4f}s")
        print(f"- Without message: {without_msg_mean:.4f}s")
        print(f"- Difference: {with_msg_mean - without_msg_mean:.4f}s")
        
    except Exception as e:
        print(f"Error processing CSV files: {e}")
        raise


if __name__ == "__main__":
    main()
