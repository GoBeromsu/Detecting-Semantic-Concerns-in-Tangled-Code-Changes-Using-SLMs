#!/usr/bin/env python3
"""
Add reasoning column to tangled_ccs_dataset.csv using OpenAI API.
Follows established patterns from openai.py with structured output and proper error handling.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import openai
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration constants
DATASET_FILE_PATH: Path = Path("data/tangled_ccs_dataset.csv")
OUTPUT_FILE_PATH: Path = Path("data/tangled_ccs_dataset_with_reasoning.csv")
OPENAI_MODEL: str = "gpt-4.1-2025-04-14"
REASONING_TEMPERATURE: float = 0.5
REQUEST_DELAY_SECONDS: float = 0.5
BATCH_SIZE: int = 10
CONCURRENT_BATCH_SIZE: int = 10

# Reasoning response schema for structured output
REASONING_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Detailed reasoning explaining the classification process and decision-making steps",
        }
    },
    "required": ["reasoning"],
    "additionalProperties": False,
}

# Structured output format for reasoning generation
REASONING_STRUCTURED_OUTPUT_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "reasoning_generation_response",
        "schema": REASONING_RESPONSE_SCHEMA,
        "strict": True,
    },
}

# System prompt for reasoning generation
SYSTEM_PROMPT: str = """You are a software engineer providing detailed reasoning and justification for pre-assigned labels on code changes from tangled commits.
Your task is to analyze the given code changes and explain why the assigned labels are appropriate based on the classification instructions.

Label definitions for reference:
- Purpose labels: the motivation behind making a code change (feat, fix, refactor)
- Object labels: the essence of the code changes that have been made (docs, test, ci, build)
     - Use an object label only when the code unit is fully dedicated to that artifact category (e.g., writing test logic, modifying documentation).

# Instructions
1. For each code unit, review the change and determine the most appropriate label from the unified set.
2. If multiple labels seem possible, resolve the overlap by applying the following rule:
     - **Purpose + Purpose**: Choose the label that best reflects *why* the change was made — `fix` if resolving a bug, `feat` if adding new capability, `refactor` if improving structure without changing behavior.
     - **Object + Object**: Choose the label that reflects the *functional role* of the artifact being modified — e.g., even if changing build logic, editing a CI script should be labeled as `ci`.
     - **Purpose + Object**: If the change is driven by code behavior (e.g., fixing test logic), assign a purpose label; if it is entirely scoped to a support artifact (e.g., adding new tests), assign an object label.
3. Repeat step 1–2 for each code unit.
4. Once all code units are labeled, return a unique set of assigned labels for the entire commit

# Reasoning Format
Your reasoning must follow exactly this numbered format:

1. [Analyze the code changes and identify what was modified or added - keep to 1-2 concise sentences]
2. [Explain how these changes align with the pre-assigned labels and their definitions - focus on direct connections]
3. [Justify why the assigned labels are appropriate and rule out alternative labels - be specific but brief]
4. [Provide final justification with clear reasoning for the label assignments - summarize in 1 sentence]

**Writing Guidelines:**
- Keep each numbered point to 1-2 sentences maximum
- Use direct, specific language - avoid unnecessary descriptive words
- Focus on the essential reasoning - eliminate redundant explanations
- Follow the instructions precisely without adding extra context

## Example
1. Added conditional check that skips queue pausing when NC_WORKER_CONTAINER='false'.
2. This prevents unwanted queue behavior - aligns with 'fix' label definition for resolving bugs.
3. 'fix' is correct because it resolves incorrect behavior, not 'feat' (no new functionality) or 'refactor' (changes behavior).
4. The 'fix' label correctly captures resolving a deployment-specific bug through a conditional safeguard.

# Labels Reference
- feat: Introduces new features to the codebase.
- fix: Fixes bugs or faults in the codebase.
- refactor: Restructures existing code without changing external behavior (e.g., improves readability, simplifies complexity, removes unused code).
- docs: Modifies documentation or text (e.g., fixes typos, updates comments or docs).
- test: Modifies test files (e.g., adds or updates tests).
- ci: Updates CI (Continuous Integration) configuration files or scripts (e.g., `.travis.yml`, `.github/workflows`).
- build: Affects the build system (e.g., updates dependencies, changes build configs or scripts)."""


def load_dataset_from_path(file_path: Path) -> pd.DataFrame:
    """Load dataset from CSV file path."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset from {file_path}: {e}")
        raise


def authenticate_openai_client() -> openai.AsyncOpenAI:
    """Authenticate with OpenAI API and return async client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    return openai.AsyncOpenAI(api_key=api_key)


def create_reasoning_prompt(commit_message: str, diff: str, types: str) -> str:
    """Create user prompt for reasoning generation."""
    try:
        types_list = json.loads(types)
        types_formatted = ", ".join(types_list)
    except json.JSONDecodeError:
        types_formatted = types

    return f"""Commit Message: {commit_message}

Assigned Types: {types_formatted}

Git Diff:
{diff}

Please generate detailed reasoning explaining how you would apply the classification instructions to analyze this tangled commit."""


async def generate_reasoning_with_openai(
    client: openai.AsyncOpenAI,
    commit_message: str,
    diff: str,
    types: str,
    row_index: int,
) -> str:
    """Generate reasoning using OpenAI API with structured output."""
    try:
        user_prompt = create_reasoning_prompt(commit_message, diff, types)

        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=REASONING_TEMPERATURE,
            response_format=REASONING_STRUCTURED_OUTPUT_FORMAT,
        )

        content = response.choices[0].message.content
        if not content:
            logging.warning(f"Empty response from OpenAI for row {row_index}")
            return "No reasoning generated"

        # Parse structured response
        try:
            response_data = json.loads(content)
            reasoning = response_data.get("reasoning", "No reasoning in response")
            logging.info(f"Generated reasoning for row {row_index}")
            return reasoning.strip()
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse structured response for row {row_index}")
            return content.strip()

    except openai.APIError as e:
        logging.error(f"OpenAI API error for row {row_index}: {e}")
        return f"API Error: {e}"
    except Exception as e:
        logging.error(f"Unexpected error for row {row_index}: {e}")
        return f"Error: {e}"


async def process_single_row_with_index(
    client: openai.AsyncOpenAI,
    idx: int,
    row: pd.Series,
) -> tuple[int, str]:
    """Process a single row and return index with result."""
    commit_message = str(row["commit_message"])
    diff = str(row["diff"])
    types = str(row["types"])

    reasoning = await generate_reasoning_with_openai(
        client, commit_message, diff, types, idx + 1
    )

    return idx, reasoning


async def process_batch_concurrently(
    client: openai.AsyncOpenAI,
    batch_rows: list,
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Process a batch of rows concurrently with safe index matching."""
    # Create tasks with index tracking
    tasks = [process_single_row_with_index(client, idx, row) for idx, row in batch_rows]

    # Execute batch concurrently - results maintain order due to asyncio.gather
    results = await asyncio.gather(*tasks)

    # Assign results back to DataFrame using returned indices
    for idx, reasoning in results:
        df.at[idx, "reason"] = reasoning
        logging.info(f"Assigned reasoning to index {idx}: {reasoning[:50]}...")

    # Save progress after batch
    save_dataset_to_path(df, output_path)


async def add_reasoning_column_to_dataset(
    df: pd.DataFrame, client: openai.AsyncOpenAI, output_path: Path
) -> pd.DataFrame:
    """Add reasoning column to dataset using OpenAI API with batch concurrent processing."""
    total_rows = len(df)

    # Initialize reason column with empty values
    df["reason"] = ""

    logging.info(f"Starting reasoning generation for {total_rows} rows...")
    logging.info(f"Concurrent batch size: {CONCURRENT_BATCH_SIZE}")

    # Process in batches for concurrent execution
    batch_rows = []
    for idx, row in df.iterrows():
        batch_rows.append((idx, row))

        # Process batch when it reaches the concurrent batch size
        if len(batch_rows) >= CONCURRENT_BATCH_SIZE:
            logging.info(f"Processing batch of {len(batch_rows)} rows concurrently...")
            await process_batch_concurrently(client, batch_rows, df, output_path)
            logging.info(f"Completed batch, saved progress")

            # Rate limiting delay between batches
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            batch_rows = []

    # Process remaining rows if any
    if batch_rows:
        logging.info(f"Processing final batch of {len(batch_rows)} rows...")
        await process_batch_concurrently(client, batch_rows, df, output_path)

    logging.info("Successfully added reasoning column to dataset")
    return df


def save_dataset_to_path(df: pd.DataFrame, output_path: Path) -> None:
    """Save dataset to CSV file path."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logging.info(f"Saved dataset to {output_path}")
        logging.info(f"Dataset shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")

    except Exception as e:
        logging.error(f"Error saving dataset to {output_path}: {e}")
        raise


def validate_environment_setup() -> None:
    """Validate environment configuration and file availability."""
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Check if dataset file exists
    if not DATASET_FILE_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE_PATH}")


def configure_logging() -> None:
    """Configure logging with structured format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


async def process_reasoning_generation(
    input_file_path: Path, output_file_path: Path, test_mode: bool = False
) -> None:
    """Process reasoning generation for dataset."""
    # Load dataset
    df = load_dataset_from_path(input_file_path)

    # Check if reason column already exists
    if "reason" in df.columns:
        if test_mode:
            logging.info("Reason column exists in test mode, dropping for regeneration")
            df = df.drop("reason", axis=1)
        else:
            logging.warning("Reason column already exists in dataset")
            response = input("Do you want to regenerate all reasoning? (y/n): ")
            if response.lower() != "y":
                logging.info("Exiting without changes")
                return
            df = df.drop("reason", axis=1)

    # Use AsyncOpenAI client with context manager (recommended by OpenAI docs)
    async with authenticate_openai_client() as client:
        # Generate reasoning with real-time saving
        df_with_reasoning = await add_reasoning_column_to_dataset(
            df, client, output_file_path
        )

    # Final save (already saved during processing, but ensure final state)
    save_dataset_to_path(df_with_reasoning, output_file_path)


def main() -> None:
    """Main function to add reasoning column to dataset."""
    configure_logging()

    logging.info("Starting reasoning column addition process...")

    try:
        # Determine if running in test mode
        test_mode = len(sys.argv) > 1 and sys.argv[1] == "--test"

        # Set dataset file path based on mode
        if test_mode:
            logging.info("Running in test mode with sample data")
            global DATASET_FILE_PATH
            DATASET_FILE_PATH = Path("../data/tangled_test_sample.csv")
            output_path = Path("../data/tangled_test_sample_with_reasoning.csv")
        else:
            output_path = OUTPUT_FILE_PATH

        # Validate environment after path is set
        validate_environment_setup()

        logging.info(f"Input: {DATASET_FILE_PATH}")
        logging.info(f"Output: {output_path}")
        logging.info(f"Model: {OPENAI_MODEL}")
        logging.info(f"Temperature: {REASONING_TEMPERATURE}")

        asyncio.run(
            process_reasoning_generation(DATASET_FILE_PATH, output_path, test_mode)
        )

        logging.info("Reasoning column addition completed successfully!")

    except Exception as e:
        logging.error(f"Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
