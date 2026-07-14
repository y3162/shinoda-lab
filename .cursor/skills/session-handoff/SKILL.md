---
name: session-handoff
description: Compress the current Cursor conversation and workspace state into a structured handoff file, or load an existing handoff to resume work in another conversation. Use when the user asks to summarize, checkpoint, hand over, resume, continue, or transfer the current development session.
---

# Session Handoff

Create and consume durable development-session handoffs.

A handoff is not a general conversation summary. It must contain enough verified information for another agent conversation to resume the work without repeating completed investigation.

## Supported modes

Infer the mode from the user's request.

* `save`: Create a complete handoff and archive the previous handoff.
* `checkpoint`: Update `LATEST.md` with the current state.
* `load`: Read and validate a handoff, but do not modify project files.
* `continue`: Read a handoff, inspect the current workspace, and resume from its next action.
* `list`: List available handoff files with date, topic, and status.

When no mode is specified:

* Use `save` when the user wants to move to another conversation.
* Use `load` when the user refers to an existing handoff.
* Use `checkpoint` when the user wants to preserve current progress.

## Storage

Use the following paths relative to the workspace root:

```text
.cursor/handoffs/LATEST.md
.cursor/handoffs/archive/<timestamp>-<slug>.md
```

Use local time for the timestamp:

```text
YYYYMMDD-HHMMSS
```

Do not write handoff data into Cursor Rules or `AGENTS.md`.

Handoffs represent temporary session state. Cursor Rules and `AGENTS.md` represent persistent instructions.

## General principles

1. Preserve decisions, evidence, constraints, and pending work.
2. Remove conversational repetition, speculation, and obsolete approaches.
3. Distinguish verified facts from assumptions.
4. Reference files by workspace-relative path.
5. Include exact commands and exact error messages only when needed to resume work.
6. Never claim that a command, test, or experiment succeeded unless its result was observed.
7. Do not copy large source files or complete file contents into the handoff.
8. Do not include passwords, API keys, access tokens, cookies, private keys, or complete environment-variable dumps.
9. Creating a handoff must not alter implementation files.
10. Keep the handoff concise enough to load into a new conversation without significant context consumption.

## Information priority

When generating a handoff, prioritize information in the following order:

1. Current workspace contents
2. Observed command and test results
3. Explicit user instructions
4. Decisions made during the conversation
5. Reasonable but unverified inferences

When the conversation conflicts with the current workspace, record the discrepancy instead of silently choosing one interpretation.

## Save procedure

### 1. Determine workspace context

Identify:

* The workspace root
* The primary project or task
* Relevant source files
* Relevant configuration files
* Relevant tests
* Generated artifacts
* Logs or outputs required for continuation

Inspect only files relevant to the current task.

Do not scan unrelated parts of a large workspace without a concrete reason.

### 2. Extract conversation state

From the current conversation, identify:

* The user's actual objective
* The current implementation or investigation state
* Decisions that have already been made
* Approaches that were attempted and rejected
* Files created or changed
* Commands executed
* Tests, metrics, logs, or observations
* Errors that remain unresolved
* Constraints and user preferences
* Questions that remain open
* The smallest concrete next action

Do not preserve casual discussion unless it changes implementation decisions.

### 3. Classify information

Every important statement must fit one of these categories:

* `VERIFIED`: Supported by inspected code, file contents, command output, test output, or explicit user confirmation.
* `INFERRED`: A reasonable interpretation that has not been directly validated.
* `UNVERIFIED`: Proposed or expected, but not checked.

Prefer verified information.

### 4. Resolve stale information

Compare conversation claims with the current workspace.

When they conflict:

* Treat the current file contents as the source of truth for implementation state.
* Preserve the conversation claim under `Discrepancies`.
* Do not silently merge contradictory information.

Examples of stale information include:

* A referenced file no longer exists
* A function was renamed
* A previously reported error is no longer reproducible
* A proposed change has already been implemented
* The handoff refers to an older configuration
* Generated outputs have been replaced

### 5. Generate the handoff

Use `assets/handoff-template.md`.

The handoff should normally remain below approximately 2,000 words. It may exceed this limit only when exact errors, migration constraints, or several independent workstreams are required for safe continuation.

The `Next action` section must be executable and specific.

Do not use vague instructions such as:

* Continue implementation
* Fix remaining issues
* Investigate further
* Finish testing
* Review the code

A valid next action names the file, command, function, experiment, or decision to address first.

### 6. Save files

Before overwriting `LATEST.md`:

1. Read the existing `LATEST.md`, if present.
2. If it represents a meaningfully different state, copy its contents into `archive/`.
3. Write the new handoff to `LATEST.md`.
4. Also save an immutable copy in `archive/`.

Do not archive an identical handoff twice.

### 7. Report completion

Report only:

* The handoff path
* Its topic
* Whether relevant workspace files were inspected
* The first next action
* Any information that could not be captured reliably

## Checkpoint procedure

Follow the save procedure with these differences:

* Set status to `in-progress`.
* Focus on changes since the previous checkpoint.
* Preserve unresolved experiments and temporary findings.
* Update `LATEST.md`.
* Archive only when there are meaningful changes.

A checkpoint may contain incomplete hypotheses, but they must be marked as `INFERRED` or `UNVERIFIED`.

## Load procedure

### 1. Select the handoff

Use, in order:

1. A path explicitly supplied by the user
2. `.cursor/handoffs/LATEST.md`
3. The newest file under `.cursor/handoffs/archive/`

When multiple handoffs appear equally relevant, present the candidates briefly instead of selecting one arbitrarily.

### 2. Read the handoff completely

Do not start modifying code before reading:

* Objective
* Resume summary
* Current state
* Decisions
* Relevant files
* Changes made
* Validation
* Errors
* Constraints
* Open tasks
* Next action
* Discrepancies
* Uncertainties

### 3. Revalidate the workspace

Confirm that:

* Referenced files still exist
* Relevant functions, classes, and configuration entries still exist
* Recorded file contents are still current
* Previously reported errors remain relevant
* Previously completed tasks have not been superseded
* The recorded next action has not already been completed

Mark the handoff as stale when:

* Referenced files no longer exist
* Relevant symbols have been renamed or removed
* Configuration has materially changed
* Previously recorded errors are no longer reproducible
* The implementation has progressed beyond the handoff
* The next action is no longer applicable

A stale handoff may still be useful, but discrepancies must be reported before continuing.

### 4. Produce a resume brief

Return a compact brief containing:

* Objective
* Loaded handoff path
* Whether the handoff is current or stale
* Completed work
* Current implementation state
* Current blocker
* First next action
* Relevant discrepancies

In `load` mode, stop after the brief.

## Continue procedure

First perform the complete load procedure.

Then:

1. Reconcile stale information.
2. Inspect the files needed for the first next action.
3. Continue from the recorded state.
4. Do not repeat completed work merely to reconstruct context.
5. Re-run validation when relevant files or configuration have changed.
6. Update the handoff after completing a meaningful unit of work.

When the first next action is already complete, select the highest-priority unfinished task and record why the handoff was advanced.

## List procedure

List handoffs newest first.

For each handoff, show:

* Timestamp
* Topic
* Status
* Primary files
* Current blocker
* First unfinished task

Do not load every file in full unless metadata is missing.

## Security and privacy

Never include:

* Secret values
* Authentication headers
* `.env` contents
* Personal communications unrelated to the engineering task
* Full production datasets
* Proprietary source code copied unnecessarily into the handoff

Secret names may be recorded when necessary, but their values must be replaced with `[REDACTED]`.

## Quality check

Before saving, verify that a new agent could answer all of the following from the handoff:

1. What is being built or investigated?
2. What has already been completed?
3. What decisions must not be reconsidered without new evidence?
4. Which files are relevant?
5. What was actually tested?
6. What currently fails or remains uncertain?
7. What should be done first?
8. What constraints must be preserved?
9. Which information is verified, inferred, or unverified?
10. Is the handoff consistent with the current workspace?

If any answer is missing, improve the handoff before saving.
