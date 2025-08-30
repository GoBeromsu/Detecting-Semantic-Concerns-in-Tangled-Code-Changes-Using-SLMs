import os
import sys
import json
from pathlib import Path
from typing import List
import tiktoken
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

# Encoding configuration
ENCODING_NAME = "cl100k_base"  # GPT-4 encoding


def _truncate_diffs_equally(
    diffs: List[str], available_tokens: int, encoding: tiktoken.Encoding
) -> str:
    """Truncate a list of diffs to fit within available_tokens, allocating equally per diff."""
    if available_tokens <= 0 or len(diffs) == 0:
        return ""

    tokens_per_diff: int = max(available_tokens // len(diffs), 0)
    tokenized: List[List[int]] = [encoding.encode(d) for d in diffs]
    truncated_tokens: List[List[int]] = [t[:tokens_per_diff] for t in tokenized]
    truncated_texts: List[str] = [encoding.decode(t) for t in truncated_tokens]
    return "\n".join(truncated_texts)


def add_truncated_commits(
    tangled_df: pd.DataFrame,
    context_window: int,
    include_message: bool,
) -> pd.DataFrame:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    truncated_texts: List[str] = []
    token_lengths: List[int] = []

    for _, row in tangled_df.iterrows():
        message: str = str(row.get("commit_message", ""))
        diffs: List[str] = json.loads(row.get("diff", "[]"))

        if include_message:
            message_tokens: List[int] = encoding.encode(message)
            remaining_tokens: int = max(context_window - len(message_tokens), 0)
            truncated_diffs: str = _truncate_diffs_equally(
                diffs, remaining_tokens, encoding
            )
            truncated_text: str = (
                f"- given commit message:\n {message}\n Diff: {truncated_diffs}"
            )
        else:
            truncated_diffs: str = _truncate_diffs_equally(
                diffs, context_window, encoding
            )
            truncated_text: str = f"- given commit diff: \n {truncated_diffs}"

        truncated_texts.append(truncated_text)
        token_lengths.append(len(encoding.encode(truncated_text)))

    return tangled_df.assign(
        truncated_commit=truncated_texts, token_length=token_lengths
    )
