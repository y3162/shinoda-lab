---
handoff_version: 3
task_id: "<short-stable-task-id>"
project: "<project>"
topic: "<short topic>"
status: "in-progress | blocked | ready-for-review | completed"
updated_at: "<ISO 8601 timestamp>"
primary_files:
  - "<project-relative path>"
---

# Resume

Continue task `<task_id>` as the next turn of the previous conversation. Treat `[V]` as established, preserve accepted decisions, and do not repeat completed work. First verify the files below, then perform **First action**. If the workspace differs, record the discrepancy and reconcile it before editing.

## Goal

- **Objective:** <concrete outcome>
- **Done when:** <acceptance condition>

## State

- **Done:** [V] <completed result>
- **Current:** [V] <implementation or investigation state>
- **Blocker:** [V] <only if applicable; change to [U] when unverified>

## Decisions

- **Keep:** [V] <decision> — <reason/evidence>
- **Do not repeat:** [V] <approach> — <observed result>

## Files

- `<path>#<symbol or section>` — <role>; <current state>; <follow-up>

## Evidence

- [V] `<command, test, metric, or observation>` → <result>
- [V] `<path or configuration>` → <relevant fact>
- **Error, if needed:** `<short exact excerpt>`

## Constraints

- [V] <technical, user, design, or scope constraint>

## Next

1. **[P0] First action:** `<file/command/symbol>` — <exact operation>
   - **Expected:** <result>
   - **Validate:** <test or observation>
   - **Stop if:** <condition requiring a decision>
2. **[P1]** <next independent task>

## Unknowns and discrepancies

- [I] <working assumption>
- [U] <question or unvalidated behavior>
- <previous claim> vs <current observation> — <status>
