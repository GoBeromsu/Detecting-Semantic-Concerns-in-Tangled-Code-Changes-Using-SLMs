"""The pinned grammar registry and its deterministic lookup tables.

``language`` is the study-internal key used by detectors and manifests;
``grammar`` is the name the Tree-sitter distribution registers. The two differ
whenever upstream spells a language differently (``jsx`` parses with the
``javascript`` grammar, ``c_sharp`` with ``csharp``), so every ``grammar`` value
is validated against the distribution registry rather than trusted by eye.
"""

from __future__ import annotations

import re
from typing import Final

from .grammar_types import FileRole, GrammarRule

RULES: Final[tuple[GrammarRule, ...]] = (
    GrammarRule("bash", "bash", FileRole.SOURCE_AST, (".bash", ".bats", ".sh")),
    GrammarRule("c", "c", FileRole.SOURCE_AST, (".c", ".h")),
    GrammarRule("cpp", "cpp", FileRole.SOURCE_AST, (".cc", ".cpp", ".cxx", ".hh", ".hpp")),
    GrammarRule("clarity", "clarity", FileRole.SOURCE_AST, (".clar",)),
    GrammarRule("css", "css", FileRole.SOURCE_AST, (".css",)),
    GrammarRule("cue", "cue", FileRole.SOURCE_AST, (".cue",)),
    GrammarRule("c_sharp", "csharp", FileRole.SOURCE_AST, (".cs",)),
    GrammarRule("dockerfile", "dockerfile", FileRole.SOURCE_AST, ()),
    GrammarRule("elixir", "elixir", FileRole.SOURCE_AST, (".ex", ".exs")),
    GrammarRule("erb", "embeddedtemplate", FileRole.SOURCE_AST, (".erb",)),
    GrammarRule("erg", "erg", FileRole.SOURCE_AST, (".er",)),
    GrammarRule("go", "go", FileRole.SOURCE_AST, (".go",)),
    GrammarRule("java", "java", FileRole.SOURCE_AST, (".java",)),
    GrammarRule("javascript", "javascript", FileRole.SOURCE_AST, (".js", ".mjs", ".cjs")),
    GrammarRule("jsx", "javascript", FileRole.SOURCE_AST, (".jsx",)),
    GrammarRule("groovy", "groovy", FileRole.SOURCE_AST, (".gradle",)),
    GrammarRule("just", "just", FileRole.SOURCE_AST, ()),
    GrammarRule("make", "make", FileRole.SOURCE_AST, ()),
    GrammarRule("nix", "nix", FileRole.SOURCE_AST, (".nix",)),
    GrammarRule("objective_c", "objc", FileRole.SOURCE_AST, (".m", ".mm")),
    GrammarRule("powershell", "powershell", FileRole.SOURCE_AST, (".ps1", ".psm1")),
    GrammarRule("pug", "pug", FileRole.SOURCE_AST, (".pug",)),
    GrammarRule("python", "python", FileRole.SOURCE_AST, (".py",)),
    GrammarRule("ruby", "ruby", FileRole.SOURCE_AST, (".rb",)),
    GrammarRule("rust", "rust", FileRole.SOURCE_AST, (".rs",)),
    GrammarRule("scss", "scss", FileRole.SOURCE_AST, (".scss",)),
    GrammarRule("sql", "sql", FileRole.SOURCE_AST, (".sql",)),
    GrammarRule("swift", "swift", FileRole.SOURCE_AST, (".swift",)),
    GrammarRule("typescript", "typescript", FileRole.SOURCE_AST, (".ts",)),
    GrammarRule("tsx", "tsx", FileRole.SOURCE_AST, (".tsx",)),
    GrammarRule("vue", "vue", FileRole.SOURCE_AST, (".vue",)),
    GrammarRule("gn", "gn", FileRole.STRUCTURED_PARSE_TREE, (".gn", ".gni")),
    GrammarRule("csv", "csv", FileRole.STRUCTURED_PARSE_TREE, (".csv",)),
    GrammarRule("hcl", "hcl", FileRole.STRUCTURED_PARSE_TREE, (".hcl", ".tf")),
    GrammarRule("html", "html", FileRole.STRUCTURED_PARSE_TREE, (".html", ".htm")),
    GrammarRule("json", "json", FileRole.STRUCTURED_PARSE_TREE, (".json",)),
    GrammarRule("markdown", "markdown", FileRole.STRUCTURED_PARSE_TREE, (".md", ".mdx", ".qmd")),
    GrammarRule("proto", "proto", FileRole.STRUCTURED_PARSE_TREE, (".proto",)),
    GrammarRule("toml", "toml", FileRole.STRUCTURED_PARSE_TREE, (".toml",)),
    GrammarRule("xml", "xml", FileRole.STRUCTURED_PARSE_TREE, (".dmn", ".xml")),
    GrammarRule("yaml", "yaml", FileRole.STRUCTURED_PARSE_TREE, (".yaml", ".yml")),
)

# Grammars this study wants but the pinned distribution does not register. Listing
# them keeps an honest `unsupported_source` distinguishable from a misspelling.
KNOWN_ABSENT_GRAMMARS: Final = frozenset({"erg"})

RULE_BY_EXTENSION: Final = {extension: rule for rule in RULES for extension in rule.extensions}
RULE_BY_LANGUAGE: Final = {rule.language: rule for rule in RULES}
BINARY_EXTENSIONS: Final = frozenset({".gif", ".ico", ".jpeg", ".jpg", ".png", ".syso", ".wasm", ".xz"})
GENERATED_EXTENSIONS: Final = frozenset({".expected", ".golden", ".lock", ".manifest", ".map", ".snap", ".sum"})
TEXT_EXTENSIONS: Final = frozenset({".ci", ".conf", ".ctl", ".dsl", ".properties", ".template", ".txt"})
EXACT_TEXT_NAMES: Final = frozenset({"CODEOWNERS", "DEPS", "LICENSE"})
EXACT_FILENAME_LANGUAGES: Final = {"Jenkinsfile": "groovy", "Makefile": "make", "justfile": "just"}
SCORE_LIMIT: Final = 4
SCORE_MARGIN: Final = 0.15
DIFF_PATH_PATTERN: Final = re.compile(r'^(?:---|\+\+\+) "?[ab]/(.+?)"?$')
CONTENT_SIGNATURES: Final = (
    (r"\bdefmodule\s+[A-Z]", ("elixir",)),
    (r"\bpublic\s+class\s+[A-Z]", ("c_sharp", "java")),
    (r"\bpackage\s+main\b", ("go",)),
    (r"\bfn\s+main\s*\(", ("rust",)),
    (r"\bsyntax\s*=\s*[\"']proto[23]", ("proto",)),
    (r"\bresource\s+[\"'][^\"']+[\"']\s+[\"']", ("hcl",)),
)


def exact_rule(name: str) -> GrammarRule | None:
    """Resolve a rule from a whole filename, before any extension lookup."""
    if name == "Dockerfile" or name.startswith("Dockerfile."):
        return RULE_BY_LANGUAGE["dockerfile"]
    language = EXACT_FILENAME_LANGUAGES.get(name)
    return RULE_BY_LANGUAGE.get(language) if language is not None else None
