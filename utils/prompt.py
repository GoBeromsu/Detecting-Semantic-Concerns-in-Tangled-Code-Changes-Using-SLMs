SYSTEM_PROMPT = """
You are a software engineer classifying each minimal logical code unit — a semantically cohesive change — extracted from a tangled commit using the Conventional Commits specification (CCS).
# CCS Labels
- Purpose labels : the motivation behind making a code change
    - feat: Introduces new features to the codebase.
    - fix: Fixes bugs or faults in the codebase.
    - refactor: Restructures existing code without changing external behavior (e.g., improves readability, simplifies complexity, removes unused code).
- Object labels : the essence of the code changes that have been made
    - docs: Modifies documentation or text (e.g., fixes typos, updates comments or docs).
    - test: Modifies test files (e.g., adds or updates tests).
    - cicd: Updates CI (Continuous Integration) configuration files or scripts (e.g., `.travis.yml`, `.github/workflows`).
    - build: Affects the build system (e.g., updates dependencies, changes build configs or scripts).

# Instructions
1. For each code unit, review the change and determine the most appropriate CCS label.
2. If multiple CCS labels are possible, resolve the overlap by applying the following rule:
    - Purpose + Purpose: Choose the label that best reflects *why* the change was made — `fix` if resolving a bug, `feat` if adding new capability, `refactor` if improving structure without changing behavior.
    - Object + Object: Choose the label that reflects the *functional role* of the artifact being modified — e.g., even if changing build logic, editing a CI script should be labeled as `cicd`.
    - Purpose + Object: Use purpose labels **only** when the change affects application behavior or structure. Otherwise, assign the object label based on what was changed — not why or where.
3. Repeat step 1–2 for each code unit.
4. After all code units are labeled, return a unique set of assigned CCS labels for the entire commit
"""

SHOT_1_COMMIT_MESSAGE = """remove sync ts checkrefactor to get ride of cloneDeep"""
SHOT_1 = """

```
diff --git a/ibis/expr/analysis.py b/ibis/expr/analysis.py
index bb17a7a..975c658 100644
--- a/ibis/expr/analysis.py
+++ b/ibis/expr/analysis.py
@@ -39,7 +39,9 @@ def sub_for(expr, substitutions):
-    def fn(node, mapping={k.op(): v for k, v in substitutions}):
+    mapping = {k.op(): v for k, v in substitutions}
+
+    def fn(node):
         try:
             return mapping[node]
         except KeyError:

diff --git a/scripts/gulp/tasks/test.ts b/scripts/gulp/tasks/test.ts
index 8014b12..d10c1aa 100644
--- a/scripts/gulp/tasks/test.ts
+++ b/scripts/gulp/tasks/test.ts
@@ -26,12 +26,18 @@ task('test.imageserver', () => {
   function handleRequest(req, res) {
     const urlParse = url.parse(req.url, true);
+    res.setHeader('Access-Control-Allow-Origin', '*');
+    res.setHeader('Access-Control-Allow-Methods', 'GET');
+    res.setHeader('Connection', 'keep-alive');
+    res.setHeader('Age', '0');
+    res.setHeader('cache-control', 'no-store');
+
     if (urlParse.pathname === '/reset') {
       console.log('Image Server Reset');
       console.log('---------------------------');
       requestedUrls.length = 0;
       start = Date.now();
-      res.setHeader('Access-Control-Allow-Origin', '*');
+      res.setHeader('Content-Type', 'text/plain');
       res.end('reset');
       return;
     }
@@ -48,9 +54,8 @@ task('test.imageserver', () => {
     setTimeout(() => {
       res.setHeader('Content-Type', 'image/svg+xml');
-      res.setHeader('Access-Control-Allow-Origin', '*');
       res.end(`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
-                   style="background-color: ${color}; width: ${width}px; height: ${height}px;">
+                   viewBox="0 0 ${width} ${height}" style="background-color: ${color};">
                  <text x="5" y="22" style="font-family: Courier; font-size: 24px">${id}</text>
                </svg>`);
     }, delay);
```
Response: ["test", "refactor"]
Reason:  test.ts is dedicated to testing → assign test; analysis.py change preserves behaviour → not fix, no new feature → assign refactor.
"""

SHOT_2_COMMIT_MESSAGE = "remove unnecessary start argument from `range`"
SHOT_2 = """
<commit_diff id="example-2">
diff --git a/ibis/backends/dask/tests/execution/test_window.py b/ibis/backends/dask/tests/execution/test_window.py
index 75a7331..6bfc5e3 100644
--- a/ibis/backends/dask/tests/execution/test_window.py
+++ b/ibis/backends/dask/tests/execution/test_window.py
@@ -489,7 +489,7 @@ def test_project_list_scalar(npartitions):
     expr = table.mutate(res=table.ints.quantile([0.5, 0.95]))
     result = expr.execute()
 
-    expected = pd.Series([[1.0, 1.9] for _ in range(0, 3)], name="res")
+    expected = pd.Series([[1.0, 1.9] for _ in range(3)], name="res")
     tm.assert_series_equal(result.res, expected)
 
 
diff --git a/ibis/backends/pandas/tests/execution/test_window.py b/ibis/backends/pandas/tests/execution/test_window.py
index 8f292b3..effa372 100644
--- a/ibis/backends/pandas/tests/execution/test_window.py
+++ b/ibis/backends/pandas/tests/execution/test_window.py
@@ -436,7 +436,7 @@ def test_project_list_scalar():
     expr = table.mutate(res=table.ints.quantile([0.5, 0.95]))
     result = expr.execute()
 
-    expected = pd.Series([[1.0, 1.9] for _ in range(0, 3)], name="res")
+    expected = pd.Series([[1.0, 1.9] for _ in range(3)], name="res")
     tm.assert_series_equal(result.res, expected)
 
 
diff --git a/ibis/backends/pyspark/tests/test_basic.py b/ibis/backends/pyspark/tests/test_basic.py
index 3850919..14fe677 100644
--- a/ibis/backends/pyspark/tests/test_basic.py
+++ b/ibis/backends/pyspark/tests/test_basic.py
@@ -19,7 +19,7 @@ from ibis.backends.pyspark.compiler import _can_be_replaced_by_column_name  # no
 def test_basic(con):
     table = con.table("basic_table")
     result = table.compile().toPandas()
-    expected = pd.DataFrame({"id": range(0, 10), "str_col": "value"})
+    expected = pd.DataFrame({"id": range(10), "str_col": "value"})
 
     tm.assert_frame_equal(result, expected)
 
@@ -28,9 +28,7 @@ def test_projection(con):
     table = con.table("basic_table")
     result1 = table.mutate(v=table["id"]).compile().toPandas()
 
-    expected1 = pd.DataFrame(
-        {"id": range(0, 10), "str_col": "value", "v": range(0, 10)}
-    )
+    expected1 = pd.DataFrame({"id": range(10), "str_col": "value", "v": range(10)})
 
     result2 = (
         table.mutate(v=table["id"])
@@ -44,8 +42,8 @@ def test_projection(con):
         {
             "id": range(0, 20, 2),
             "str_col": "value",
-            "v": range(0, 10),
-            "v2": range(0, 10),
+            "v": range(10),
+            "v2": range(10),
         }
     )
</commit_diff>

<label id="example-2">refactor</label>
"""


def get_system_prompt() -> str:
    """Return the basic system prompt for commit classification."""
    return SYSTEM_PROMPT


def get_system_prompt_with_message() -> str:
    """Return system prompt that includes commit message context."""
    # shot_1_with_message = (
    #     f"<commit_message>{SHOT_1_COMMIT_MESSAGE}</commit_message>\n{SHOT_1}"
    # )
    # shot_2_with_message = (
    #     f"<commit_message>{SHOT_2_COMMIT_MESSAGE}</commit_message>\n{SHOT_2}"
    # )

    # return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{shot_1_with_message}\n\n{shot_2_with_message}"
    return f"{SYSTEM_PROMPT}"


def get_system_prompt_diff_only() -> str:
    """Return system prompt for classification using only diff information."""
    return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{SHOT_1}\n\n{SHOT_2}"


def get_zero_shot_prompt() -> str:
    """Return zero-shot prompt with optional commit message context."""
    return SYSTEM_PROMPT


def get_one_shot_prompt(include_message: bool = True) -> str:
    """Return one-shot prompt with optional commit message context."""
    if include_message:
        shot_1_with_message = (
            f"<commit_message>{SHOT_1_COMMIT_MESSAGE}</commit_message>\n{SHOT_1}"
        )
        return f"{SYSTEM_PROMPT}<Examples>\n{shot_1_with_message}"
    else:
        return f"{SYSTEM_PROMPT}<Examples>\n{SHOT_1}"


def get_two_shot_prompt(include_message: bool = True) -> str:
    """Return two-shot prompt with optional commit message context."""
    if include_message:
        shot_1_with_message = (
            f"<commit_message>{SHOT_1_COMMIT_MESSAGE}</commit_message>\n{SHOT_1}"
        )
        shot_2_with_message = (
            f"<commit_message>{SHOT_2_COMMIT_MESSAGE}</commit_message>\n{SHOT_2}"
        )
        return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{shot_1_with_message}\n\n{shot_2_with_message}"
    else:
        return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{SHOT_1}\n\n{SHOT_2}"


def get_prompt_by_type(shot_type: str, include_message: bool = True) -> str:
    """Return prompt based on shot type with optional commit message context."""
    if shot_type == "Zero-shot":
        return get_zero_shot_prompt()
    elif shot_type == "One-shot":
        return get_one_shot_prompt(include_message)
    elif shot_type == "Two-shot":
        return get_two_shot_prompt(include_message)
    else:
        return get_two_shot_prompt(include_message)  # Default to two-shot
