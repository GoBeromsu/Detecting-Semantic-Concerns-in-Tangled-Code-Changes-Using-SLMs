SYSTEM_PROMPT = """
You are a software engineer classifying individual code units extracted from a tangled commit.
Each change unit (e.g., function, method, class, or code block) represents a reviewable atomic change, and must be assigned exactly one label.

Label selection must assign exactly one concern from the following unified set:
- Purpose labels : the motivation behind making a code change (feat, fix, refactor)
- Object labels : the essence of the code changes that have been made(docs, test, cicd, build)
     - Use an object label only when the code unit is fully dedicated to that artifact category (e.g., writing test logic, modifying documentation).

# Instructions
1. For each code unit, review the change and determine the most appropriate label from the unified set.
2. If multiple labels seem possible, resolve the overlap by applying the following rule:
     - **Purpose + Purpose**: Choose the label that best reflects *why* the change was made — `fix` if resolving a bug, `feat` if adding new capability, `refactor` if improving structure without changing behavior.
     - **Object + Object**: Choose the label that reflects the *functional role* of the artifact being modified — e.g., even if changing build logic, editing a CI script should be labeled as `cicd`.
     - **Purpose + Object**: If the change is driven by code behavior (e.g., fixing test logic), assign a purpose label; if it is entirely scoped to a support artifact (e.g., adding new tests), assign an object label.
3. Repeat step 1–2 for each code unit.
4. Once all code units are labeled, return a unique set of assigned labels for the entire commit

# Labels
- feat: Introduces new features to the codebase.
- fix: Fixes bugs or faults in the codebase.
- refactor: Restructures existing code without changing external behavior (e.g., improves readability, simplifies complexity, removes unused code).
- docs: Modifies documentation or text (e.g., fixes typos, updates comments or docs).
- test: Modifies test files (e.g., adds or updates tests).
- cicd: Updates CI (Continuous Integration) configuration files or scripts (e.g., `.travis.yml`, `.github/workflows`).
- build: Affects the build system (e.g., updates dependencies, changes build configs or scripts).
"""

SHOT_1_COMMIT_MESSAGE = """reintroduce timeout for assertion

The timeout had been removed by a previous commit. Without the timeout the test might be flaky.
Also removed obsolete code"""
SHOT_1 = """
<commit_diff id="example-1">
diff --git a/engine/src/test/java/io/camunda/zeebe/engine/processing/streamprocessor/StreamProcessorReplayModeTest.java b/engine/src/test/java/io/camunda/zeebe/engine/processing/streamprocessor/StreamProcessorReplayModeTest.java
index d0ee4f3..c2ab83c 100644
--- a/engine/src/test/java/io/camunda/zeebe/engine/processing/streamprocessor/StreamProcessorReplayModeTest.java
+++ b/engine/src/test/java/io/camunda/zeebe/engine/processing/streamprocessor/StreamProcessorReplayModeTest.java
@@ -13,6 +13,7 @@ import static io.camunda.zeebe.protocol.record.intent.ProcessInstanceIntent.ACTI
 import static io.camunda.zeebe.protocol.record.intent.ProcessInstanceIntent.ELEMENT_ACTIVATING;
 import static java.util.function.Predicate.isEqual;
 import static org.assertj.core.api.Assertions.assertThat;
+import static org.awaitility.Awaitility.await;
 import static org.mockito.ArgumentMatchers.any;
 import static org.mockito.ArgumentMatchers.anyLong;
 import static org.mockito.ArgumentMatchers.eq;
@@ -30,7 +31,6 @@ import io.camunda.zeebe.protocol.record.intent.ProcessInstanceIntent;
 import io.camunda.zeebe.streamprocessor.StreamProcessor;
 import io.camunda.zeebe.streamprocessor.StreamProcessor.Phase;
 import io.camunda.zeebe.streamprocessor.StreamProcessorMode;
-import org.awaitility.Awaitility;
 import org.junit.Rule;
 import org.junit.Test;
 import org.mockito.InOrder;
@@ -71,7 +71,7 @@ public final class StreamProcessorReplayModeTest {
     // when
     startStreamProcessor(replayUntilEnd);
 
-    Awaitility.await()
+    await()
         .untilAsserted(
             () -> assertThat(getCurrentPhase(replayUntilEnd)).isEqualTo(Phase.PROCESSING));
 
@@ -163,7 +163,7 @@ public final class StreamProcessorReplayModeTest {
         command().processInstance(ACTIVATE_ELEMENT, RECORD),
         event().processInstance(ELEMENT_ACTIVATING, RECORD).causedBy(0));
 
-    Awaitility.await("should have replayed first events")
+    await("should have replayed first events")
         .until(replayContinuously::getLastSuccessfulProcessedRecordPosition, (pos) -> pos > 0);
 
     // when
@@ -210,7 +210,7 @@ public final class StreamProcessorReplayModeTest {
         command().processInstance(ACTIVATE_ELEMENT, RECORD),
         event().processInstance(ELEMENT_ACTIVATING, RECORD).causedBy(0));
 
-    Awaitility.await("should have replayed first events")
+    await("should have replayed first events")
         .until(replayContinuously::getLastSuccessfulProcessedRecordPosition, (pos) -> pos > 0);
     streamProcessor.pauseProcessing().join();
     replayContinuously.writeBatch(
@@ -244,7 +244,7 @@ public final class StreamProcessorReplayModeTest {
     // then
     verify(eventApplier, TIMEOUT).applyState(anyLong(), eq(ELEMENT_ACTIVATING), any());
 
-    Awaitility.await()
+    await()
         .untilAsserted(
             () -> {
               final var lastProcessedPosition = getLastProcessedPosition(replayContinuously);
@@ -273,8 +273,7 @@ public final class StreamProcessorReplayModeTest {
 
     verify(eventApplier, TIMEOUT).applyState(anyLong(), eq(ELEMENT_ACTIVATING), any());
 
-    Awaitility.await()
-        .until(() -> getLastProcessedPosition(replayContinuously), isEqual(commandPosition));
+    await().until(() -> getLastProcessedPosition(replayContinuously), isEqual(commandPosition));
 
     // then
     assertThat(replayContinuously.getLastSuccessfulProcessedRecordPosition())
@@ -285,7 +284,6 @@ public final class StreamProcessorReplayModeTest {
   @Test
   public void shouldNotSetLastProcessedPositionIfLessThanSnapshotPosition() {
     // given
-    final var commandPositionBeforeSnapshot = 1L;
     final var snapshotPosition = 2L;
 
     startStreamProcessor(replayContinuously);
@@ -298,23 +296,20 @@ public final class StreamProcessorReplayModeTest {
     // when
     startStreamProcessor(replayContinuously);
 
-    Awaitility.await()
+    await()
         .untilAsserted(
             () -> assertThat(getCurrentPhase(replayContinuously)).isEqualTo(Phase.REPLAY));
 
-    final var eventPosition =
-        replayContinuously.writeEvent(
-            ELEMENT_ACTIVATING,
-            RECORD,
-            writer -> writer.sourceRecordPosition(commandPositionBeforeSnapshot));
-
     // then
     final var lastProcessedPositionState = replayContinuously.getLastProcessedPositionState();
 
-    assertThat(lastProcessedPositionState.getLastSuccessfulProcessedRecordPosition())
-        .describedAs(
-            "Expected that the last processed position is not less than the snapshot position")
-        .isEqualTo(snapshotPosition);
+    await()
+        .untilAsserted(
+            () ->
+                assertThat(lastProcessedPositionState.getLastSuccessfulProcessedRecordPosition())
+                    .describedAs(
+                        "Expected that the last processed position is not less than the snapshot position")
+                    .isEqualTo(snapshotPosition));
   }
 
   private StreamProcessor startStreamProcessor(final StreamProcessorRule streamProcessorRule) {</commit_diff>

<label id="example-1">refactor</label>
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
        return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{shot_1_with_message}"
    else:
        return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{SHOT_1}"


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
