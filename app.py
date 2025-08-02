import os
import json
import streamlit as st
import pandas as pd
from typing import Dict, Any

from dotenv import load_dotenv

from utils import llms
from utils.prompt import get_system_prompt_with_message, get_prompt_by_type
from utils.llms.constant import (
    CODE_DIFF_INPUT_HEIGHT,
    SYSTEM_PROMPT_INPUT_HEIGHT,
)
from visual_eval.ui.components import (
    render_evaluation_metrics,
    render_results_table,
    create_column_config,
)
from visual_eval.ui.dataset import (
    get_available_datasets,
    load_dataset,
    DIFF_COLUMN,
    TYPES_COLUMN,
    SHAS_COLUMN,
)
from visual_eval.ui.session import (
    get_model_name,
    get_api_provider,
    set_evaluation_results,
)
from visual_eval.ui.setup import render_api_setup_sidebar
from utils.eval import calculate_metrics

# Direct analysis result columns
ANALYSIS_RESULT_COLUMNS = [
    "Predicted_Concern_Types",
]

# Evaluation result columns
EVALUATION_RESULT_COLUMNS = [
    "Test_Index",
    "Predicted_Types",
    "Actual_Types",
    "Status",
    "Case_Precision",
    "Case_Recall",
    "Case_F1",
    "SHAs",
]

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


def render_system_prompt_input(title: str = "System Prompt") -> str:
    """Render system prompt input widget with consistent styling."""
    st.subheader(title)
    return st.text_area(
        "Modify the system prompt:",
        value=get_system_prompt_with_message(),
        height=SYSTEM_PROMPT_INPUT_HEIGHT,
    )


def calculate_evaluation_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics using pre-computed case metrics."""
    total_cases = len(results_df)
    if total_cases == 0:
        return {
            "total": 0,
            "type_precision": 0.0,
            "type_recall": 0.0,
            "type_f1_macro": 0.0,
        }

    # Macro-average: simply average the pre-calculated case metrics
    macro_precision = results_df["Case_Precision"].mean()
    macro_recall = results_df["Case_Recall"].mean()
    macro_f1 = results_df["Case_F1"].mean()

    return {
        "total": total_cases,
        "type_precision": macro_precision,
        "type_recall": macro_recall,
        "type_f1_macro": macro_f1,
    }


def process_single_case(row: pd.Series, system_prompt: str) -> Dict[str, Any]:
    """Core logic: Process single evaluation case with model prediction."""
    try:
        # Extract data from row
        diff = row[DIFF_COLUMN]
        actual_concern_types = (
            json.loads(row[TYPES_COLUMN]) if row[TYPES_COLUMN] else []
        )
        shas = json.loads(row[SHAS_COLUMN]) if row[SHAS_COLUMN] else []

        # Get model prediction
        model_name = get_model_name()
        provider = get_api_provider()
        predicted_concern_types = llms.api_call(
            provider=provider,
            model_name=model_name,
            commit=diff,
            system_prompt=system_prompt,
            api_key=OPENAI_KEY,
        )

        return {
            "predicted_types": predicted_concern_types,
            "actual_types": actual_concern_types,
            "shas": shas,
            "success": True,
        }
    except Exception as e:
        return {
            "predicted_types": [],
            "actual_types": json.loads(row[TYPES_COLUMN]) if row[TYPES_COLUMN] else [],
            "shas": json.loads(row[SHAS_COLUMN]) if row[SHAS_COLUMN] else [],
            "success": False,
        }


def execute_batch_concern_evaluation(df: pd.DataFrame, system_prompt: str) -> None:
    """Execute batch evaluation using streamlit and pandas delegation."""
    if df.empty:
        st.error("No test data available for evaluation")
        return

    # Debug: Show prompt info
    examples_count = system_prompt.count("<commit_diff")
    has_commit_msg = "<commit_message>" in system_prompt
    st.write(
        f"📋 **Prompt Details:** {examples_count} examples, commit messages: {'✅' if has_commit_msg else '❌'}"
    )

    # Create containers for stable rendering
    status_container = st.container()
    results_container = st.container()

    # Process all cases using pandas delegation
    with status_container:
        with st.status("Running batch evaluation...", expanded=True) as status:
            st.write(f"Processing {len(df)} test cases...")

            # Use pandas apply for batch processing
            results = []
            progress_bar = st.progress(0)

            for i, (_, row) in enumerate(df.iterrows()):
                case_result = process_single_case(row, system_prompt)
                metrics = calculate_metrics(
                    case_result["predicted_types"], case_result["actual_types"]
                )

                # Combine results
                results.append(
                    {
                        "Test_Index": i + 1,
                        "Predicted_Types": (
                            ", ".join(sorted(case_result["predicted_types"]))
                            if case_result["predicted_types"]
                            else "None"
                        ),
                        "Actual_Types": (
                            ", ".join(sorted(case_result["actual_types"]))
                            if case_result["actual_types"]
                            else "None"
                        ),
                        "Status": "Match" if metrics["exact_match"] else "No Match",
                        "Case_Precision": metrics["precision"],
                        "Case_Recall": metrics["recall"],
                        "Case_F1": metrics["f1"],
                        "SHAs": (
                            ", ".join(case_result["shas"])
                            if case_result["shas"]
                            else "None"
                        ),
                    }
                )

                progress_bar.progress((i + 1) / len(df))

                # Show warning for failed cases
                if not case_result["success"]:
                    st.warning(f"⚠️ API error for case {i+1}")

            # Create results DataFrame using pandas
            evaluation_results_df = pd.DataFrame(results)
            status.update(label="Evaluation complete!", state="complete")

    # Display results in stable container
    with results_container:
        metrics = calculate_evaluation_metrics(evaluation_results_df)
        render_evaluation_metrics(metrics, len(df))
        render_results_table(evaluation_results_df)

        # Store and download results
        # set_evaluation_results(evaluation_results_df) #TODO : dost it need?

        if not evaluation_results_df.empty:
            download_df = evaluation_results_df.drop(
                columns=["Case_Precision", "Case_Recall", "Case_F1"]
            )
            csv_data = download_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_data,
                file_name=f"evaluation_results_{len(download_df)}_cases.csv",
                mime="text/csv",
                use_container_width=True,
            )


def show_direct_input() -> None:
    """Render UI interface for direct code diff input and concern analysis."""
    # Get global prompt settings
    shot_type = st.session_state.get("global_shot_type", "Two-shot")
    include_message = st.session_state.get("global_include_message", True)
    system_prompt = get_prompt_by_type(shot_type, include_message)

    diff = st.text_area(
        "📝 Code Diff",
        placeholder="Paste the output of `git diff` here...",
        height=CODE_DIFF_INPUT_HEIGHT,
        help="Copy and paste your git diff output for analysis",
    )

    submitted = st.button("🎯 Analyze Diff", type="primary", use_container_width=True)

    if submitted and diff.strip():
        st.divider()
        st.header("📊 Analysis Results")
        with st.spinner("Analyzing code diff..."):
            model_name = get_model_name()
            provider = get_api_provider()
            print(f"Model name: {model_name}")
            predicted_concern_types = llms.api_call(
                provider=provider,
                model_name=model_name,
                commit=diff,
                system_prompt=system_prompt,
                api_key=OPENAI_KEY,
            )

            st.subheader("Concern Classification Results")
            analysis_results_df = pd.DataFrame(
                {
                    "Predicted_Concern_Types": [
                        (
                            ", ".join(sorted(predicted_concern_types))
                            if predicted_concern_types
                            else "None"
                        )
                    ],
                }
            )
            st.dataframe(
                analysis_results_df,
                use_container_width=True,
                hide_index=True,
                column_config=create_column_config(ANALYSIS_RESULT_COLUMNS),
            )
    elif submitted and not diff.strip():
        st.warning("Please enter a code diff to analyze.")


def show_csv_input() -> None:
    """Render UI interface for batch evaluation from test dataset files."""
    available_dataset_files = get_available_datasets()

    if not available_dataset_files:
        st.error("No CSV test dataset files found in datasets directory")
        st.stop()

    # Get global prompt settings
    shot_type = st.session_state.get("global_shot_type", "Two-shot")
    include_message = st.session_state.get("global_include_message", True)
    system_prompt = get_prompt_by_type(shot_type, include_message)

    selected_dataset = st.selectbox(
        "📊 Test Dataset",
        available_dataset_files,
        format_func=lambda x: f"📄 {os.path.basename(x)} ({os.path.dirname(x)})",
    )

    submitted = st.button(
        "🚀 Start Evaluation", type="primary", use_container_width=True
    )

    if submitted:
        print("shot_type", shot_type, "include_message", include_message)
        test_dataset = load_dataset(selected_dataset)
        if not test_dataset.empty:
            st.success(
                f"📊 Loaded **{len(test_dataset)}** test cases from `{selected_dataset}`"
            )
            st.divider()
            st.header("📊 Evaluation Results")
            execute_batch_concern_evaluation(test_dataset, system_prompt)
        else:
            st.error("❌ Failed to load test dataset")


def main() -> None:
    """Main application entry point for concern classification evaluation."""
    st.title("Concern is All You Need")
    load_dotenv()

    # Setup in sidebar
    with st.sidebar:
        setup_success = render_api_setup_sidebar()
        if not setup_success:
            st.stop()

    # Main content
    st.header("📝 Concern Classification Analysis")

    # Global prompt configuration
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            shot_type = st.selectbox(
                "🎯 Prompt Method",
                ["Zero-shot", "One-shot", "Two-shot"],
                index=2,
                key="global_shot_type",
            )
        with col2:
            include_message = st.toggle(
                "📝 Commit Messages", value=True, key="global_include_message"
            )

        # Global prompt preview
        system_prompt = get_prompt_by_type(shot_type, include_message)
        with st.expander("🔍 Preview Prompt", expanded=False):
            st.code(system_prompt, language="text")

    direct_input_tab, batch_evaluation_tab = st.tabs(
        ["🔍 Direct Analysis", "📊 Batch Evaluation"]
    )

    with direct_input_tab:
        show_direct_input()
    with batch_evaluation_tab:
        show_csv_input()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Concern is All You Need",
        page_icon="🌩️",
        layout="wide",
    )
    main()
