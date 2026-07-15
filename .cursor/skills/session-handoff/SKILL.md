---
name: session-handoff
description: Create or resume compact, evidence-backed handoffs for multiple tasks within each project. Use when the user asks to hand off, save a session, checkpoint progress, continue from a previous chat, resume, list handoffs, or read a handoff.
---

# Session handoff

Treat each handoff as executable continuation context for one task, not as a conversation transcript. Preserve the information that prevents the next agent from repeating work.

## Operations

- **Save**: create or update the current project's handoff. Treat `handoff`, `checkpoint`, and `save this session` as save requests.
- **Resume**: read a handoff, reconcile it with the workspace, and continue. Treat `load`, `continue`, and `resume` as resume requests.
- **List**: read the project's index and show available task IDs, status, topic, blocker, and first action without loading every handoff.
- If the user only asks to inspect a handoff, read it and return a brief without editing implementation files.

## Location

Find the repository root containing `.cursor/`. Resolve the project from the files and task currently in scope. Store handoffs under:

```text
<repo-root>/<project>/docs/handoffs/INDEX.md
<repo-root>/<project>/docs/handoffs/<task-id>.md
```

For a repository rooted at `master/`, this must be `master/<project>/docs/handoffs/<task-id>.md`. If one task spans multiple projects with separate state, write one task file and index entry per project. Ask before choosing when the project is ambiguous. Do not use `.cursor/handoffs/`.

## Task ID

- Use a short, stable, lowercase kebab-case ID such as `grpo-seq2seq` or `tokenizer-interface`.
- Reuse an existing ID only when the topic and primary files clearly match. Never overwrite another task merely because the project is the same.
- If the user supplies an ID or path, use it. Otherwise match `INDEX.md`; create a new ID from the objective when no existing task matches.
- If multiple tasks are plausible, show their IDs and ask which one to use.

## Save

1. Resolve the task ID and read its existing `<task-id>.md`, if present. Retain still-relevant unfinished work; do not create an archive or duplicate history unless requested.
2. Inspect only files relevant to the task. Use current file contents and observed command/test output as the source of truth.
3. Extract: objective and acceptance condition, completed work, current work, decisions and reasons, approaches that should not be repeated, relevant files/symbols, validation and metrics, active errors, constraints, open tasks, and one executable first action.
4. Mark facts with `[V]` (observed or explicitly confirmed), `[I]` (inferred), or `[U]` (not yet verified). Record discrepancies instead of silently resolving them.
5. Use `assets/handoff-template.md`, remove empty sections and comments, and keep the result normally within 300–800 words. Expand only when exact diagnostics or independent workstreams are needed for safe continuation.
6. Write `<task-id>.md` and upsert one compact row for it in `INDEX.md`; creating a handoff must not modify implementation files.

Do not paste source files or long logs. Include an exact command, error excerpt, metric, or configuration value only when it changes the next agent's action. Never include secrets, tokens, cookies, private keys, `.env` values, or unrelated personal information.

The first action must name a file, symbol, command, experiment, or decision. Include the expected result and how to validate it; avoid `continue`, `investigate`, or `finish testing` without a concrete target.

## Resume

1. Use a user-supplied path or task ID. Without one, read `INDEX.md`: use the only active task, or ask the user to choose when several active tasks exist.
2. Read the whole selected handoff before editing. Check that its project, relevant files, symbols, configuration, and first action still match the workspace.
3. Start with a compact brief: loaded path, task ID, current/stale status, completed work, blocker, and first action.
4. Do not repeat completed work. If the workspace diverges, report the discrepancy, reconcile it, and then select the highest-priority unfinished task.
5. Begin the first action immediately unless the user asked only to load/show the handoff. After a meaningful unit of work, update the selected task file and its `INDEX.md` row.

## Index

Keep `INDEX.md` to one row per task:

```markdown
| ID | Topic | Status | Updated | Blocker / first action |
|---|---|---|---|---|
| `grpo-seq2seq` | GRPO Seq2Seq | blocked | 2026-07-15 | inspect `scripts/grpo/loss.py` |
```

The index is for selection, not detailed context. Read the task file for the full state.

## Legacy migration

If an older `LATEST.md` exists, treat it as a legacy handoff. Assign or ask for a task ID, copy its relevant state into `<task-id>.md`, and add the task to `INDEX.md`; do not continue using one global `LATEST.md` for new saves.

## Output

After saving, report the task ID, path, project, files inspected, and first action only. After resuming, report the task ID, loaded path, whether it is current, the active blocker, and the action being started. Do not echo the entire handoff unless asked.
