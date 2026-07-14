---
handoff_version: 1
topic: "<short topic>"
status: "in-progress | blocked | ready-for-review | completed"
created_at: "<ISO 8601 timestamp>"
updated_at: "<ISO 8601 timestamp>"
workspace: "<workspace or project name>"
primary_files:
* "<workspace-relative path>"
---

# Objective

Describe the concrete outcome being pursued.

The objective should state the desired result, not merely the current activity.

Bad:

> Investigating the tokenizer.

Good:

> Implement a common tokenizer interface that supports character, unigram, and SentencePiece tokenizers with a consistent save and load format.

# Resume summary

Provide a compact description of the current state for an agent opening a new conversation.

Include:

* What is being implemented or investigated
* What has been completed
* What currently blocks progress
* What should happen next

# Workspace state

* Workspace root:
* Relevant directories:
* Primary source files:
* Relevant configuration files:
* Relevant test files:
* Relevant generated files:
* Temporary files or outputs:

# Current state

## Completed

* `[VERIFIED]` Completed work and observed results.

## In progress

* `[VERIFIED]` Partially completed work.
* `[INFERRED]` State inferred from the conversation or code.

## Blocked

* `[VERIFIED]` Current blocker.
* `[UNVERIFIED]` Suspected cause, when applicable.

# Decisions

Record decisions that affect future work.

## Accepted

* Decision:

  * Reason:
  * Evidence:
  * Consequence:

## Rejected approaches

* Approach:

  * Why it was rejected:
  * Conditions under which it should be reconsidered:

Do not include approaches that were merely mentioned but never seriously considered.

# Relevant files

* `path/to/file`

  * Purpose:
  * Relevant symbols or sections:
  * Current state:
  * Required follow-up:

Do not paste entire file contents.

# Changes made

* `path/to/file`

  * Change:
  * Reason:
  * Validation:
  * Remaining concern:

Only include changes relevant to the current objective.

# Validation

## Commands executed

```bash
# Exact commands that were actually executed
```

## Observed results

* `[VERIFIED]` Test result, metric, log observation, or failure.

## Not yet validated

* `[UNVERIFIED]` Behavior that still requires checking.

Do not describe a test as successful unless its output was observed.

# Errors and diagnostics

## Active errors

```text
Exact error excerpt required for continuation.
```

* Trigger:
* Suspected cause:
* Evidence:
* Attempts already made:
* Current hypothesis:
* Next diagnostic step:

## Resolved errors

* Error:
* Resolution:
* Verification:

# Constraints

## Technical constraints

* Runtime, dependency, API, architecture, or compatibility constraints.

## User requirements

* Explicit requirements given by the user.

## Design constraints

* Decisions that should remain stable unless new evidence appears.

## Scope boundaries

* Work that is intentionally outside the current task.

# Open tasks

Use ordered priority.

1. `[P0]` Immediate blocker
2. `[P1]` Required implementation or validation
3. `[P2]` Optional improvement

Each task should be independently understandable.

# Next action

Provide one concrete first action.

* File or command:
* Target symbol or section:
* Exact operation:
* Expected result:
* How to validate:
* Stop condition:

The next action must be small enough to begin immediately.

# Discrepancies

Record conflicts between the conversation, handoff, and current workspace.

* Previous claim:
* Current observation:
* Likely explanation:
* Resolution status:

# Uncertainties

* `[INFERRED]` Assumptions still being used
* `[UNVERIFIED]` Questions requiring investigation

# Minimal restart prompt

Continue the work described in this handoff.

First inspect the files listed under “Relevant files” and confirm that the recorded state still matches the current workspace.

Do not repeat completed work.

Start with “Next action” unless the workspace has diverged. When it has diverged, reconcile the discrepancy before modifying implementation files.
