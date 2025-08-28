#!/usr/bin/env python3
"""
Efficiency Analysis: Multiple Regression with Concern Count and Input Tokens
Analyzes the relationship between concern count, input tokens, and inference time using multiple regression.
Model: log(time) = β₀ + β₁log(tokens) + β₂concern_count + β₃(log(tokens)×concern_count) + ε
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import argparse

# Constants - Use root results directory (from project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up from RQ/analysis/ to project root
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"
OUTLIER_THRESHOLD_IQR = 1.5  # IQR multiplier for outlier detection
LOG_EPSILON = 1e-6  # Small value to avoid log(0)

# Design constants for consistent styling
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'background': '#F5F5F5',
    'text': '#2C3E50',
    'interaction': '#6A4C93',
    'concern': '#FF6B35'
}

PLOT_STYLE = {
    'figure_size': (12, 8),
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
    """Load raw data from CSV files for multiple regression analysis.
    
    Returns:
        Tuple of (cleaned_dataframe, outlier_info)
    """
    all_data = []
    
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ['context_len', 'inference_time', 'concern_count']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")
        
        df['source_file'] = csv_path.name
        all_data.append(df[['context_len', 'inference_time', 'concern_count', 'source_file']])
    
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    # Remove zero or negative values before log transformation
    original_len = len(combined_df)
    combined_df = combined_df[
        (combined_df['context_len'] > 0) & 
        (combined_df['inference_time'] > 0) &
        (combined_df['concern_count'] >= 0)
    ].copy()
    
    if len(combined_df) < original_len:
        print(f"Removed {original_len - len(combined_df)} rows with zero/negative values for log transformation")
    
    # Apply log transformations
    combined_df['log_tokens'] = np.log(combined_df['context_len'] + LOG_EPSILON)
    combined_df['log_time'] = np.log(combined_df['inference_time'] + LOG_EPSILON)
    
    # Create interaction term
    combined_df['log_tokens_x_concern'] = combined_df['log_tokens'] * combined_df['concern_count']
    
    # Detect outliers in log_time
    outlier_indices = detect_outliers_iqr(combined_df['log_time'])
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


def perform_multiple_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform multiple regression analysis with interaction term.
    
    Model: log(time) = β₀ + β₁log(tokens) + β₂concern_count + β₃(log(tokens)×concern_count) + ε
    """
    # Prepare features
    X = df[['log_tokens', 'concern_count', 'log_tokens_x_concern']].values
    y = df['log_time'].values
    
    # Fit multiple regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate predictions and metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate adjusted R-squared
    n = len(df)
    p = X.shape[1]  # number of predictors
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return {
        'intercept': float(model.intercept_),
        'coefficients': {
            'log_tokens': float(model.coef_[0]),
            'concern_count': float(model.coef_[1]),
            'interaction': float(model.coef_[2])
        },
        'model_metrics': {
            'r_squared': float(r2),
            'adjusted_r_squared': float(adj_r2),
            'rmse': float(rmse),
            'mse': float(mse)
        },
        'model': model,
        'predictions': y_pred,
        'residuals': y - y_pred
    }



def create_concern_effect_by_token_facets(df: pd.DataFrame, regression_results: Dict[str, Any], output_dir: Path) -> Path:
    """Create single plot showing concern effect on inference time with different lines for each token length.
    
    Shows how concern count affects inference time when token length is fixed (5 lines).
    """
    setup_plot_style()
    
    # Get unique values and sort them
    unique_tokens = sorted(df['context_len'].unique())
    unique_concerns = sorted(df['concern_count'].unique())
    
    # Create single plot
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figure_size'])
    
    # Create color palette for token lengths
    base_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success'], COLORS['interaction']]
    if len(unique_tokens) <= len(base_colors):
        token_colors = base_colors[:len(unique_tokens)]
    else:
        token_colors = [base_colors[i % len(base_colors)] for i in range(len(unique_tokens))]
    
    # Create marker styles for better distinction
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, token_len in enumerate(unique_tokens):
        color = token_colors[i]
        marker = markers[i % len(markers)]
        
        # Generate model predictions for this token length using fitted model
        concern_range = np.array(unique_concerns)
        log_token_fixed = np.log(token_len + LOG_EPSILON)
        
        # Prepare input features for prediction
        X_pred = np.column_stack([
            np.full(len(concern_range), log_token_fixed),  # log_tokens (fixed)
            concern_range,  # concern_count (varying)
            np.full(len(concern_range), log_token_fixed) * concern_range  # interaction
        ])
        
        # Use fitted model to predict log(time), then convert to time
        model = regression_results['model']
        y_pred = model.predict(X_pred)
        time_pred = np.exp(y_pred)
        
        # Plot prediction line
        ax.plot(concern_range, time_pred, color=color, 
               linewidth=PLOT_STYLE['line_width'], marker=marker, markersize=8,
               label=f'Input Size: {int(token_len)}', markeredgecolor=COLORS['text'], markeredgewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Number of Code Concerns', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Processing Time (seconds)', fontweight='bold', color=COLORS['text'])
    ax.set_title('How Code Complexity Affects Processing Time\n(Different Input Sizes)', 
                fontweight='bold', color=COLORS['text'], pad=20)
    ax.set_xticks(unique_concerns)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    
    # Add model information
    r2 = regression_results['model_metrics']['r_squared']
    concern_coef = regression_results['coefficients']['concern_count']
    interaction_coef = regression_results['coefficients']['interaction']
    
    stats_text = f'Concern Effect β₂ = {concern_coef:.6f}\n'
    stats_text += f'Interaction β₃ = {interaction_coef:.6f}\n'
    stats_text += f'Model R² = {r2:.3f}\n'
    stats_text += f'n = {len(df)}'
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='bottom', horizontalalignment='right', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['background'], 
                     alpha=0.9, edgecolor=COLORS['primary'], linewidth=1))
    
    plt.tight_layout()
    
    output_path = output_dir / "concern_effect_by_token_lines.png"
    plt.savefig(output_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


def create_token_effect_by_concern_facets(df: pd.DataFrame, regression_results: Dict[str, Any], output_dir: Path) -> Path:
    """Create single plot showing token effect on inference time with different lines for each concern count.
    
    Shows how token length affects inference time when concern count is fixed (5 lines).
    """
    setup_plot_style()
    
    # Get unique values and sort them
    unique_tokens = sorted(df['context_len'].unique())
    unique_concerns = sorted(df['concern_count'].unique())
    
    # Create single plot
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figure_size'])
    
    # Create color palette for concern counts
    base_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success'], COLORS['interaction']]
    if len(unique_concerns) <= len(base_colors):
        concern_colors = base_colors[:len(unique_concerns)]
    else:
        concern_colors = [base_colors[i % len(base_colors)] for i in range(len(unique_concerns))]
    
    # Create marker styles for better distinction
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, concern_count in enumerate(unique_concerns):
        color = concern_colors[i]
        marker = markers[i % len(markers)]
        
        # Generate model predictions for this concern count using fitted model
        log_tokens = np.array([np.log(t + LOG_EPSILON) for t in unique_tokens])
        
        # Prepare input features for prediction
        X_pred = np.column_stack([
            log_tokens,  # log_tokens (varying)
            np.full(len(log_tokens), concern_count),  # concern_count (fixed)
            log_tokens * concern_count  # interaction
        ])
        
        # Use fitted model to predict log(time), then convert to time
        model = regression_results['model']
        y_pred = model.predict(X_pred)
        time_pred = np.exp(y_pred)
        
        # Plot prediction line with token positions (categorical x-axis)
        positions = range(1, len(unique_tokens) + 1)
        ax.plot(positions, time_pred, color=color, 
               linewidth=PLOT_STYLE['line_width'], marker=marker, markersize=8,
               label=f'Complexity Level: {int(concern_count)}', 
               markeredgecolor=COLORS['text'], markeredgewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Input Size (tokens)', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Processing Time (seconds)', fontweight='bold', color=COLORS['text'])
    ax.set_title('How Input Size Affects Processing Time\n(Different Complexity Levels)', 
                fontweight='bold', color=COLORS['text'], pad=20)
    
    # Set token length labels on x-axis
    ax.set_xticks(range(1, len(unique_tokens) + 1))
    ax.set_xticklabels([str(int(t)) for t in unique_tokens])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    
    # Add model information
    r2 = regression_results['model_metrics']['r_squared']
    token_coef = regression_results['coefficients']['log_tokens']
    interaction_coef = regression_results['coefficients']['interaction']
    
    stats_text = f'Token Effect β₁ = {token_coef:.6f}\n'
    stats_text += f'Interaction β₃ = {interaction_coef:.6f}\n'
    stats_text += f'Model R² = {r2:.3f}\n'
    stats_text += f'n = {len(df)}'
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='bottom', horizontalalignment='right', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['background'], 
                     alpha=0.9, edgecolor=COLORS['secondary'], linewidth=1))
    
    plt.tight_layout()
    
    output_path = output_dir / "token_effect_by_concern_lines.png"
    plt.savefig(output_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


def generate_summary_stats(df: pd.DataFrame, regression_results: Dict[str, Any], outlier_info: dict) -> dict:
    """Generate comprehensive summary statistics for the multiple regression analysis."""
    
    # Calculate correlations
    correlations = {
        'tokens_time': df['context_len'].corr(df['inference_time']),
        'concern_time': df['concern_count'].corr(df['inference_time']),
        'tokens_concern': df['context_len'].corr(df['concern_count']),
        'log_tokens_log_time': df['log_tokens'].corr(df['log_time']),
        'concern_log_time': df['concern_count'].corr(df['log_time'])
    }
    
    stats_dict = {
        'sample_size': len(df),
        'outliers_detected': outlier_info['count'],
        'outliers_removed': outlier_info['removed'],
        'descriptive_stats': {
            'context_len': {
                'mean': df['context_len'].mean(),
                'std': df['context_len'].std(),
                'min': df['context_len'].min(),
                'max': df['context_len'].max()
            },
            'inference_time': {
                'mean': df['inference_time'].mean(),
                'std': df['inference_time'].std(),
                'min': df['inference_time'].min(),
                'max': df['inference_time'].max()
            },
            'concern_count': {
                'mean': df['concern_count'].mean(),
                'std': df['concern_count'].std(),
                'min': df['concern_count'].min(),
                'max': df['concern_count'].max()
            }
        },
        'correlations': correlations,
        'regression_results': regression_results
    }
    
    return stats_dict


def save_summary_json(stats_dict: dict, outlier_info: dict, csv_files: List[str], output_dir: Path) -> Path:
    """Save comprehensive summary as JSON."""
    from datetime import datetime, timezone
    import json
    
    regression_results = stats_dict['regression_results']
    
    # Build comprehensive summary
    summary = {
        "analysis_info": {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "analysis_type": "efficiency_concern_count_input_token_regression",
            "model_formula": "log(time) = β₀ + β₁log(tokens) + β₂concern_count + β₃(log(tokens)×concern_count) + ε",
            "input_files": csv_files,
            "outlier_detection_method": "IQR on log(time)",
            "outlier_threshold": f"{OUTLIER_THRESHOLD_IQR}x IQR"
        },
        "data_summary": {
            "total_samples": int(stats_dict['sample_size']),
            "outliers_detected": int(stats_dict['outliers_detected']),
            "outliers_removed": bool(stats_dict['outliers_removed'])
        },
        "regression_model": {
            "coefficients": {
                "intercept": {
                    "value": regression_results['intercept'],
                    "interpretation": "log(time) when log(tokens)=0 and concern_count=0"
                },
                "log_tokens": {
                    "value": regression_results['coefficients']['log_tokens'],
                    "interpretation": "β₁: Effect of log(tokens) when concern_count is fixed"
                },
                "concern_count": {
                    "value": regression_results['coefficients']['concern_count'],
                    "interpretation": "β₂: Effect of concern_count when log(tokens) is fixed"
                },
                "interaction": {
                    "value": regression_results['coefficients']['interaction'],
                    "interpretation": "β₃: How concern_count modifies the effect of log(tokens)"
                }
            },
            "model_fit": {
                "r_squared": regression_results['model_metrics']['r_squared'],
                "adjusted_r_squared": regression_results['model_metrics']['adjusted_r_squared'],
                "rmse": regression_results['model_metrics']['rmse'],
                "interpretation": (
                    "Excellent fit" if regression_results['model_metrics']['r_squared'] >= 0.9 else
                    "Good fit" if regression_results['model_metrics']['r_squared'] >= 0.7 else
                    "Moderate fit" if regression_results['model_metrics']['r_squared'] >= 0.5 else
                    "Weak fit" if regression_results['model_metrics']['r_squared'] >= 0.3 else
                    "Poor fit"
                )
            }
        },
        "correlation_analysis": {
            "pearson_correlations": {
                "tokens_time": float(stats_dict['correlations']['tokens_time']),
                "concern_time": float(stats_dict['correlations']['concern_time']),
                "tokens_concern": float(stats_dict['correlations']['tokens_concern']),
                "log_tokens_log_time": float(stats_dict['correlations']['log_tokens_log_time']),
                "concern_log_time": float(stats_dict['correlations']['concern_log_time'])
            },
            "mechanically_linked": {
                "tokens_concern_correlation": float(stats_dict['correlations']['tokens_concern']),
                "interpretation": "Strong correlation indicates mechanical linkage between variables"
            }
        },
        "descriptive_statistics": {
            "context_len": {
                "mean": float(stats_dict['descriptive_stats']['context_len']['mean']),
                "std": float(stats_dict['descriptive_stats']['context_len']['std']),
                "min": int(stats_dict['descriptive_stats']['context_len']['min']),
                "max": int(stats_dict['descriptive_stats']['context_len']['max']),
                "unit": "tokens"
            },
            "inference_time": {
                "mean": float(stats_dict['descriptive_stats']['inference_time']['mean']),
                "std": float(stats_dict['descriptive_stats']['inference_time']['std']),
                "min": float(stats_dict['descriptive_stats']['inference_time']['min']),
                "max": float(stats_dict['descriptive_stats']['inference_time']['max']),
                "unit": "seconds"
            },
            "concern_count": {
                "mean": float(stats_dict['descriptive_stats']['concern_count']['mean']),
                "std": float(stats_dict['descriptive_stats']['concern_count']['std']),
                "min": int(stats_dict['descriptive_stats']['concern_count']['min']),
                "max": int(stats_dict['descriptive_stats']['concern_count']['max']),
                "unit": "count"
            }
        },
        "outlier_analysis": {
            "detection_method": "Interquartile Range (IQR) on log(time)",
            "threshold": f"Q1 - {OUTLIER_THRESHOLD_IQR}×IQR and Q3 + {OUTLIER_THRESHOLD_IQR}×IQR",
            "total_detected": int(outlier_info['count']),
            "outlier_values": [round(v, 2) for v in outlier_info['values']] if outlier_info['values'] else [],
            "impact": {
                "removed_from_analysis": bool(outlier_info['removed']),
                "percentage_of_data": round((int(outlier_info['count']) / (int(stats_dict['sample_size']) + int(outlier_info['count']))) * 100, 1) if outlier_info['count'] > 0 else 0
            }
        }
    }
    
    output_path = output_dir / "efficiency_concern_count_input_token_summary.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Multiple regression analysis: concern count, input tokens vs inference time')
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
        expected_columns = {'context_len', 'inference_time', 'concern_count'}
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
        output_dir = ANALYSIS_OUTPUT_DIR / f"cc_it_{files_summary}{outlier_suffix}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting efficiency analysis: Concern Count + Input Tokens vs Inference Time")
    print("Model: log(time) = β₀ + β₁log(tokens) + β₂concern_count + β₃(log(tokens)×concern_count) + ε")
    print("=" * 80)
    
    # Process CSV files
    print(f"Processing {len(args.csv_files)} CSV file(s)...")
    csv_paths = [Path(f) for f in args.csv_files]
    
    try:
        # Load data with outlier handling
        remove_outliers = not args.keep_outliers
        df, outlier_info = load_csv_data(csv_paths, remove_outliers=remove_outliers)
        print(f"Loaded {len(df)} data points from CSV files")
        
        if outlier_info['count'] > 0:
            print(f"Detected {outlier_info['count']} outliers in log(inference_time)")
            if outlier_info['removed']:
                print(f"Outliers removed from analysis")
            else:
                print(f"Outliers kept in analysis (use --keep-outliers flag)")
        else:
            print("No outliers detected in log(inference_time)")
        
        # Perform multiple regression analysis
        print(f"\nPerforming multiple regression analysis...")
        regression_results = perform_multiple_regression(df)
        
        # Generate statistics
        stats = generate_summary_stats(df, regression_results, outlier_info)
        
        # Print results
        print(f"\nMultiple Regression Results:")
        print(f"- Sample size: {stats['sample_size']}")
        print(f"- R² = {regression_results['model_metrics']['r_squared']:.4f}")
        print(f"- Adjusted R² = {regression_results['model_metrics']['adjusted_r_squared']:.4f}")
        print(f"- RMSE = {regression_results['model_metrics']['rmse']:.4f}")
        
        print(f"\nCoefficients:")
        print(f"- Intercept (β₀): {regression_results['intercept']:.4f}")
        print(f"- log(tokens) (β₁): {regression_results['coefficients']['log_tokens']:.6f}")
        print(f"- concern_count (β₂): {regression_results['coefficients']['concern_count']:.6f}")
        print(f"- interaction (β₃): {regression_results['coefficients']['interaction']:.6f}")
        
        print(f"\nCorrelations:")
        print(f"- tokens ↔ time: {stats['correlations']['tokens_time']:.3f}")
        print(f"- concern ↔ time: {stats['correlations']['concern_time']:.3f}")
        print(f"- tokens ↔ concern: {stats['correlations']['tokens_concern']:.3f}")
        
        # Generate visualizations
        print(f"\nGenerating visualizations...")
        concern_facets_path = create_concern_effect_by_token_facets(df, regression_results, output_dir)
        token_facets_path = create_token_effect_by_concern_facets(df, regression_results, output_dir)
        
        # Save results
        json_path = save_summary_json(stats, outlier_info, args.csv_files, output_dir)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"- Concern effect lines: {concern_facets_path.name}")
        print(f"- Token effect lines: {token_facets_path.name}")
        print(f"- Summary JSON: {json_path.name}")
        
        # Final model equation
        β0, β1, β2, β3 = (regression_results['intercept'], 
                          regression_results['coefficients']['log_tokens'],
                          regression_results['coefficients']['concern_count'],
                          regression_results['coefficients']['interaction'])
        print(f"\nFinal model equation:")
        print(f"log(time) = {β0:.4f} + {β1:.6f}×log(tokens) + {β2:.6f}×concern_count + {β3:.6f}×(log(tokens)×concern_count)")
        
    except Exception as e:
        print(f"Error processing CSV files: {e}")
        raise


if __name__ == "__main__":
    main()
