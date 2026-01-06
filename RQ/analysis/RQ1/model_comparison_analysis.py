#!/usr/bin/env python3
"""
Model Comparison Analysis: LLM vs SLM vs Fine-tuned SLM
Compares performance between GPT-4.1, Qwen, and Qwen Fine-tuned models.
Analyzes cases where models disagree and their prediction patterns.
"""

import pandas as pd
import json
import ast
import yaml
from pathlib import Path
import argparse
from datetime import datetime

from . import PROJECT_ROOT, ANALYSIS_OUTPUT_DIR, CONFIG_PATH


def load_config():
    """Load configuration from config.yaml"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_json_types(type_str: str) -> list:
    """Parse JSON string containing types array."""
    try:
        # Handle string representation of list
        if isinstance(type_str, str):
            return ast.literal_eval(type_str)
        return type_str
    except (ValueError, SyntaxError):
        # Fallback: try to parse as JSON
        try:
            return json.loads(type_str)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse types: {type_str}")
            return []


def check_any_label_match(predicted_types: list, actual_types: list) -> bool:
    """Check if any predicted label matches any actual label."""
    if not predicted_types or not actual_types:
        return False
    return bool(set(predicted_types) & set(actual_types))


def load_model_results(csv_path: Path, model_name: str) -> pd.DataFrame:
    """Load and preprocess model results from CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse types columns
    df["predicted_types_parsed"] = df["predicted_types"].apply(parse_json_types)
    df["actual_types_parsed"] = df["actual_types"].apply(parse_json_types)

    # Calculate both exact match and any label match
    df[f"{model_name}_any_match"] = df.apply(
        lambda row: check_any_label_match(
            row["predicted_types_parsed"], row["actual_types_parsed"]
        ),
        axis=1,
    )

    # Select relevant columns for this model
    result_df = pd.DataFrame(
        {
            f"{model_name}_predicted": df["predicted_types_parsed"],
            f"{model_name}_exact_match": df["exact_match"],
            f"{model_name}_any_match": df[f"{model_name}_any_match"],
            "actual_types": df["actual_types_parsed"],  # Keep one copy of actual types
        }
    )

    return result_df


def create_comparison_dataset(
    llm_df: pd.DataFrame, slm_df: pd.DataFrame, ft_slm_df: pd.DataFrame
) -> pd.DataFrame:
    """Create combined dataset with all model predictions and match results."""

    # Combine all model results by index (all CSV files have same order)
    dataset = pd.DataFrame(
        {
            "Row_Index": range(len(llm_df)),
            "Actual_Types": llm_df["actual_types"].apply(str),
            "LLM_Prediction": llm_df["LLM_predicted"].apply(str),
            "LLM_Exact_Match": llm_df["LLM_exact_match"],
            "LLM_Any_Match": llm_df["LLM_any_match"],
            "SLM_Prediction": slm_df["SLM_predicted"].apply(str),
            "SLM_Exact_Match": slm_df["SLM_exact_match"],
            "SLM_Any_Match": slm_df["SLM_any_match"],
            "FineTuned_SLM_Prediction": ft_slm_df["Fine_tuned_SLM_predicted"].apply(
                str
            ),
            "FineTuned_SLM_Exact_Match": ft_slm_df["Fine_tuned_SLM_exact_match"],
            "FineTuned_SLM_Any_Match": ft_slm_df["Fine_tuned_SLM_any_match"],
        }
    )

    return dataset


def analyze_performance_patterns(dataset: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance patterns using pandas operations."""

    # Add performance analysis columns
    analysis = dataset.copy()

    # Conservative analysis: models that got NO labels right (any_match = False)
    analysis["LLM_Complete_Miss"] = ~analysis["LLM_Any_Match"]
    analysis["SLM_Complete_Miss"] = ~analysis["SLM_Any_Match"]
    analysis["FineTuned_SLM_Complete_Miss"] = ~analysis["FineTuned_SLM_Any_Match"]

    # Count how many models completely missed
    analysis["Total_Complete_Misses"] = (
        analysis["LLM_Complete_Miss"].astype(int)
        + analysis["SLM_Complete_Miss"].astype(int)
        + analysis["FineTuned_SLM_Complete_Miss"].astype(int)
    )

    return analysis


def generate_comprehensive_analysis(analysis: pd.DataFrame) -> dict:
    """Generate comprehensive JSON analysis using pandas vectorized operations."""

    total_samples = len(analysis)

    # Complete miss statistics using pandas vectorized operations
    miss_cols = [
        "LLM_Complete_Miss",
        "SLM_Complete_Miss",
        "FineTuned_SLM_Complete_Miss",
    ]
    miss_counts = analysis[miss_cols].sum()
    complete_miss_stats = {
        "total_samples": total_samples,
        **{
            col.replace("_Complete_Miss", ""): {
                "complete_misses": int(miss_counts[col]),
                "percentage": float(miss_counts[col] / total_samples * 100),
            }
            for col in miss_cols
        },
    }

    # Extract specific patterns using pandas boolean indexing
    all_models_miss = analysis[analysis["Total_Complete_Misses"] == 3]
    llm_miss_others_correct = analysis[
        analysis["LLM_Complete_Miss"]
        & ~analysis["SLM_Complete_Miss"]
        & ~analysis["FineTuned_SLM_Complete_Miss"]
    ]
    slms_miss_llm_correct = analysis[
        ~analysis["LLM_Complete_Miss"]
        & analysis["SLM_Complete_Miss"]
        & analysis["FineTuned_SLM_Complete_Miss"]
    ]

    # Convert dataframes to examples using pandas to_dict
    def df_to_examples(df: pd.DataFrame) -> list:
        return (
            df[
                [
                    "Actual_Types",
                    "LLM_Prediction",
                    "SLM_Prediction",
                    "FineTuned_SLM_Prediction",
                ]
            ]
            .rename(
                columns={
                    "Actual_Types": "actual_types",
                    "LLM_Prediction": "LLM_prediction",
                    "SLM_Prediction": "SLM_prediction",
                    "FineTuned_SLM_Prediction": "FineTuned_SLM_prediction",
                }
            )
            .to_dict("records")
        )

    # Generate difficulty rankings using pandas vectorized operations
    def get_difficulty_ranking(model_column: str) -> list:
        """Get difficulty ranking using pandas vectorized operations."""
        model_misses = analysis[analysis[model_column]]
        # Parse actual types and create sorted tuples for consistent grouping
        parsed_types = (
            model_misses["Actual_Types"]
            .apply(ast.literal_eval)
            .apply(lambda x: tuple(sorted(x)))
        )
        # Use pandas value_counts instead of Counter
        miss_counts = parsed_types.value_counts()
        total_model_misses = len(model_misses)

        return [
            {
                "label_combination": list(label_combo),
                "miss_count": int(count),
                "percentage_of_model_misses": float(count / total_model_misses * 100),
            }
            for label_combo, count in miss_counts.items()
        ]

    # Generate miss pattern distribution using pandas value_counts
    miss_pattern_counts = analysis["Total_Complete_Misses"].value_counts().sort_index()
    miss_pattern_analysis = {
        f"{miss_count}_models_missed": {
            "count": int(sample_count),
            "percentage": float(sample_count / total_samples * 100),
        }
        for miss_count, sample_count in miss_pattern_counts.items()
    }

    return {
        "analysis_metadata": {
            "total_samples": total_samples,
            "analysis_timestamp": datetime.now().isoformat(),
            "description": "Conservative analysis of complete misses using pandas vectorized operations",
        },
        "complete_miss_statistics": complete_miss_stats,
        "miss_pattern_distribution": miss_pattern_analysis,
        "common_failures": {
            "all_models_missed": {
                "count": len(all_models_miss),
                "percentage": float(len(all_models_miss) / total_samples * 100),
                "examples": df_to_examples(all_models_miss),
            }
        },
        "unique_model_patterns": {
            "llm_unique_failures": {
                "description": "LLM missed but both SLMs got it right",
                "count": len(llm_miss_others_correct),
                "percentage": float(len(llm_miss_others_correct) / total_samples * 100),
                "examples": df_to_examples(llm_miss_others_correct),
            },
            "llm_unique_successes": {
                "description": "Both SLMs missed but LLM got it right",
                "count": len(slms_miss_llm_correct),
                "percentage": float(len(slms_miss_llm_correct) / total_samples * 100),
                "examples": df_to_examples(slms_miss_llm_correct),
            },
        },
        "difficulty_rankings": {
            "LLM": get_difficulty_ranking("LLM_Complete_Miss"),
            "SLM": get_difficulty_ranking("SLM_Complete_Miss"),
            "FineTuned_SLM": get_difficulty_ranking("FineTuned_SLM_Complete_Miss"),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare performance between LLM, SLM, and Fine-tuned SLM models"
    )
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--use-config",
        action="store_true",
        default=True,
        help="Use config.yaml for model configuration",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()
    script_config = config["rq1"]["scripts"]["model_comparison_analysis"]

    print("📊 Loading model results...")

    # Load model results from config
    model_results = {}
    model_paths = {}

    for model_name, model_config in script_config["models"].items():
        csv_path = PROJECT_ROOT / model_config["csv_path"]
        model_paths[model_name] = csv_path

        print(f"  {model_name}: {csv_path}")

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load results with simplified model names for processing
        simple_name = model_name.replace(" ", "_").replace("-", "_")
        model_results[simple_name] = load_model_results(csv_path, simple_name)

    # Extract the specific model DataFrames
    llm_df = model_results["LLM"]
    slm_df = model_results["SLM"]
    ft_slm_df = model_results["Fine_tuned_SLM"]

    print(
        f"✅ Loaded results: LLM({len(llm_df)}), SLM({len(slm_df)}), Fine-tuned SLM({len(ft_slm_df)})"
    )

    # Create comparison dataset
    dataset = create_comparison_dataset(llm_df, slm_df, ft_slm_df)
    print(f"📈 Generated dataset with {len(dataset)} samples")

    # Analyze performance patterns
    analysis = analyze_performance_patterns(dataset)
    print(f"📊 Added performance analysis columns")

    # Generate comprehensive JSON analysis
    comprehensive_analysis = generate_comprehensive_analysis(analysis)
    print(f"🔍 Generated comprehensive analysis")

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ANALYSIS_OUTPUT_DIR / "pf_model_comparison"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    dataset_csv_path = output_dir / "model_comparison_dataset.csv"
    analysis_json_path = output_dir / "model_comparison_analysis.json"

    dataset.to_csv(dataset_csv_path, index=False)

    # Save comprehensive analysis as JSON
    with open(analysis_json_path, "w", encoding="utf-8") as f:
        json.dump(comprehensive_analysis, f, indent=2, ensure_ascii=False)

    # Print key findings from JSON analysis
    print("\n=== 🎯 COMPREHENSIVE ANALYSIS SUMMARY ===")

    # 1. Complete Miss Statistics
    stats = comprehensive_analysis["complete_miss_statistics"]
    print(f"\n📊 Complete Miss Statistics (out of {stats['total_samples']} samples):")
    print(
        f"  LLM: {stats['LLM']['complete_misses']} ({stats['LLM']['percentage']:.1f}%) - Best performer"
    )
    print(
        f"  SLM: {stats['SLM']['complete_misses']} ({stats['SLM']['percentage']:.1f}%) - Worst performer"
    )
    print(
        f"  Fine-tuned SLM: {stats['FineTuned_SLM']['complete_misses']} ({stats['FineTuned_SLM']['percentage']:.1f}%) - Middle performer"
    )

    # 2. Common failures
    common = comprehensive_analysis["common_failures"]["all_models_missed"]
    print(
        f"\n🔴 All 3 Models Failed: {common['count']} cases ({common['percentage']:.1f}%)"
    )
    if common["examples"]:
        print("  Examples:")
        for i, example in enumerate(common["examples"][:3]):
            print(f"    {i+1}. Actual: {example['actual_types']}")
            print(
                f"       LLM→{example['LLM_prediction']}, SLM→{example['SLM_prediction']}, FT→{example['FineTuned_SLM_prediction']}"
            )

    # 3. Unique patterns
    unique = comprehensive_analysis["unique_model_patterns"]
    llm_failures = unique["llm_unique_failures"]
    llm_successes = unique["llm_unique_successes"]

    print(f"\n🎯 Model-Specific Patterns:")
    print(
        f"  LLM failed, SLMs succeeded: {llm_failures['count']} cases ({llm_failures['percentage']:.1f}%)"
    )
    print(
        f"  SLMs failed, LLM succeeded: {llm_successes['count']} cases ({llm_successes['percentage']:.1f}%)"
    )

    # 4. Difficulty rankings (top 3 for each model)
    print(f"\n🏆 Most Difficult Label Combinations (Top 3 per model):")
    rankings = comprehensive_analysis["difficulty_rankings"]

    for model_name in ["LLM", "SLM", "FineTuned_SLM"]:
        print(f"  {model_name}:")
        model_ranking = rankings[model_name][:3]
        for i, entry in enumerate(model_ranking):
            labels = entry["label_combination"]
            count = entry["miss_count"]
            print(f"    {i+1}. {labels} ({count} misses)")

    print(f"\n💾 Results saved to:")
    print(f"  📄 Dataset (CSV): {dataset_csv_path}")
    print(f"  📋 Analysis (JSON): {analysis_json_path}")
    print(
        f"\n🔍 For detailed analysis, check the JSON file with complete data and examples."
    )


if __name__ == "__main__":
    main()
