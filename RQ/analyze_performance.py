import pandas as pd
from pathlib import Path


def load_results() -> pd.DataFrame:
    """Load results from CSV file"""
    results_path = Path("results/analysis/results_table.csv")
    return pd.read_csv(results_path)


def analyze_context_length_vs_inference_time(df: pd.DataFrame) -> None:
    """Analyze relationship between context length and inference time"""
    print("=== Context Length vs Inference Time Analysis ===")

    # Group by model and context length
    grouped = (
        df.groupby(["Model", "Context Length"])["Inference Time"].mean().reset_index()
    )

    for model in grouped["Model"].unique():
        model_data = grouped[grouped["Model"] == model]
        print(f"\n{model}:")
        for _, row in model_data.iterrows():
            print(f"  {row['Context Length']:5d}: {row['Inference Time']:.2f}s")


def analyze_accuracy_vs_context_length(df: pd.DataFrame) -> None:
    """Analyze relationship between accuracy and context length"""
    print("\n=== Accuracy vs Context Length Analysis ===")

    # Group by model and context length
    grouped = df.groupby(["Model", "Context Length"])["Accuracy"].mean().reset_index()

    for model in grouped["Model"].unique():
        model_data = grouped[grouped["Model"] == model]
        print(f"\n{model}:")
        for _, row in model_data.iterrows():
            print(f"  {row['Context Length']:5d}: {row['Accuracy']:.3f}")


def analyze_message_impact(df: pd.DataFrame) -> None:
    """Analyze impact of 'With Message' on performance"""
    print("\n=== Message Impact Analysis ===")

    summary = (
        df.groupby(["Model", "With Message"])
        .agg({"Accuracy": "mean", "Inference Time": "mean"})
        .round(3)
    )

    print(summary)


def generate_correlations(df: pd.DataFrame) -> None:
    """Generate correlation analysis"""
    print("\n=== Correlation Analysis ===")

    # Calculate correlations for each model
    for model in df["Model"].unique():
        model_data = df[df["Model"] == model]

        context_time_corr = model_data["Context Length"].corr(
            model_data["Inference Time"]
        )
        context_acc_corr = model_data["Context Length"].corr(model_data["Accuracy"])

        print(f"\n{model}:")
        print(f"  Context Length ↔ Inference Time: {context_time_corr:.3f}")
        print(f"  Context Length ↔ Accuracy:       {context_acc_corr:.3f}")


def main() -> None:
    """Main analysis function"""
    df = load_results()

    analyze_context_length_vs_inference_time(df)
    analyze_accuracy_vs_context_length(df)
    analyze_message_impact(df)
    generate_correlations(df)


if __name__ == "__main__":
    main()
