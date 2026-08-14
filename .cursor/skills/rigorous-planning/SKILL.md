---
name: rigorous-planning
description: Critically evaluate feasibility, eliminate material ambiguity through structured multiple-choice questions, and produce implementation-ready software plans. Use for non-trivial implementation or architectural tasks in Cursor Plan Mode, especially when requirements, scope, behavior, architecture, dependencies, interfaces, or acceptance criteria may be underspecified.
---

# Rigorous Planning

Use this skill to turn a user's implementation request into a critically evaluated, unambiguous, and executable plan.

The goal is not to agree with the user's proposed solution. The goal is to determine whether the requested outcome can be achieved, resolve every material decision that belongs to the user, and produce the strongest practical plan supported by the available evidence.

## Core Principles

1. Never resolve a material ambiguity by guessing.
2. Distinguish facts that can be discovered from decisions that require user intent.
3. Treat the user's proposed implementation as a proposal to evaluate, not as an automatically correct design.
4. Prefer evidence from the actual codebase, configuration, documentation, dependencies, and runtime constraints over inference.
5. Do not produce an implementation plan that contains unresolved material decisions.
6. Prefer solutions that are general, valid, feasible, and proportionate to the problem.
7. Do not silently replace a questionable user instruction with your preferred design. Explain the issue and ask the user to choose.
8. While operating in Plan Mode, plan and investigate only. Do not implement the plan until the user explicitly proceeds to the build or execution phase.

## Material Ambiguity

An ambiguity is material if different answers could reasonably change any of the following:

- externally observable behavior;
- scope or acceptance criteria;
- public or internal interfaces;
- data models or persistence;
- architecture or component boundaries;
- dependency choices;
- compatibility requirements;
- security or privacy behavior;
- failure handling;
- performance characteristics;
- migration or rollout behavior;
- testing strategy;
- destructive or irreversible operations;
- files or systems that must be modified.

Do not ask the user about facts that can be reliably determined from the repository or environment. Investigate those facts instead.

Do not invent missing user preferences.

If a detail is neither discoverable nor material to the resulting implementation, it does not need to block planning.

## Workflow

Follow these phases in order.

### Phase 1: Understand the Request

Extract and separate:

- desired outcome;
- explicit requirements;
- explicit constraints;
- user-proposed implementation choices;
- acceptance criteria;
- known environment constraints;
- missing information.

Do not treat a proposed implementation method as a requirement unless the user clearly states that it is mandatory.

Identify contradictions between requirements before proceeding.

### Phase 2: Investigate the Existing System

Before asking questions that the repository can answer, inspect the relevant codebase and configuration.

Determine, where applicable:

- existing architecture and component boundaries;
- analogous implementations;
- repository conventions;
- dependency and framework versions;
- APIs and data structures involved;
- existing tests and validation mechanisms;
- technical constraints imposed by the current system;
- likely files and code paths affected.

Use read-only investigation during planning.

Do not modify files, install dependencies, change configuration, create migrations, or perform other state-changing actions merely to determine feasibility.

### Phase 3: Feasibility Gate

Evaluate whether the requested outcome is technically achievable under the stated constraints.

Classify the request as one of:

#### Feasible

There is a credible implementation path using the available environment and constraints.

Proceed to the clarification phase.

#### Conditionally Feasible

The outcome is feasible only if specific prerequisites, constraint changes, dependencies, permissions, migrations, or external conditions are accepted.

State those conditions explicitly and ask the user to decide where necessary.

#### Infeasible

No valid implementation path exists under the stated constraints.

Stop planning the requested implementation.

Report:

- the blocking constraint;
- the evidence for the conclusion;
- why reasonable alternatives do not satisfy the stated requirements;
- what requirement or constraint would have to change to make the task feasible, if applicable.

Do not fabricate a plan for an infeasible request.

#### Feasibility Undetermined

A user-controlled decision or unavailable fact prevents a reliable feasibility determination.

Ask only the question or questions necessary to resolve feasibility, then repeat this phase.

Never classify a task as feasible merely because an implementation seems plausible.

### Phase 4: Critical Design Review

For feasible or conditionally feasible requests, critically evaluate the user's requested approach.

Check for:

- unnecessary coupling;
- duplicated functionality;
- one-off special cases;
- incompatibility with existing abstractions;
- hidden migration requirements;
- security or correctness risks;
- unnecessary dependencies;
- excessive implementation complexity;
- poor testability;
- avoidable performance costs;
- conflict with established repository conventions;
- a simpler or more general solution that satisfies the same requirements.

Maintain critical neutrality.

Do not defend the user's approach merely because the user proposed it.

Do not reject it merely because another design is stylistically preferable.

Criticism must be tied to concrete consequences, constraints, or evidence.

### Phase 5: Clarification Loop

Before producing the final plan, resolve every remaining material ambiguity.

Use the user's answer for user-owned decisions.

Use repository evidence for discoverable technical facts.

Repeat the clarification process until no material ambiguity remains.

#### Question Format

Whenever user input is required, use a multiple-choice question.

The first option must be the recommended choice.

Every question must include an `Other` option.

Use this structure:

```text
<Decision that must be made>

A. [Recommended] <recommended choice>
   <brief reason this is recommended>

B. <alternative choice>
   <important trade-off>

C. <another choice, if materially useful>
   <important trade-off>

D. Other
   Please specify.
````

For independent decisions, multiple questions may be asked together if doing so does not make their consequences harder to understand.

Do not ask vague questions such as:

* "What do you prefer?"
* "How should I implement this?"
* "Should I proceed?"
* "Is this okay?"

Instead, identify the actual decision and present concrete alternatives.

Do not use an unstated default if the user does not answer a material question.

If the user's answer introduces a new material ambiguity, ask another multiple-choice question.

If the user explicitly delegates a specific decision to you, choose the recommended option based on the available evidence and record that delegation as a resolved decision.

### Phase 6: Challenge Weak User Constraints

Before finalizing the plan, verify that the user's requested constraints and implementation choices allow a solution with sufficient:

* generality;
* validity;
* feasibility.

Also prefer proportional complexity: do not introduce architecture that is substantially more complex than the problem requires.

#### Generality

The design should solve the stated class of problem rather than relying on unnecessary one-off cases.

Prefer existing reusable abstractions when they fit.

Do not over-generalize beyond plausible requirements.

#### Validity

The design must satisfy the confirmed requirements and preserve relevant system invariants.

It must be consistent with the actual architecture and interfaces.

#### Feasibility

Each planned step must be implementable and verifiable in the actual repository and environment, or have an explicit prerequisite.

If the user's requested approach materially fails one of these criteria but remains technically possible, do not silently accept or replace it.

Explain the concern and ask for confirmation using the required multiple-choice format.

The recommended option should normally be the corrective approach.

A non-recommended option may preserve the user's original approach if it remains genuinely feasible, but its consequences must be stated clearly.

If the approach is actually infeasible, return to the Feasibility Gate instead.

### Phase 7: Produce the Plan

Only produce the final implementation plan when:

* feasibility has been established;
* all material ambiguities have been resolved;
* conflicting requirements have been resolved;
* problematic but feasible user constraints have been explicitly confirmed;
* the proposed approach has passed the generality, validity, and feasibility checks.

The plan must be implementation-ready.

Include the following sections when relevant:

### Objective

State the exact outcome to be implemented.

### Confirmed Requirements and Constraints

Record the requirements that the implementation must satisfy.

Do not include inferred preferences as requirements.

### Feasibility Assessment

State why the implementation is feasible and list any confirmed prerequisites.

### Resolved Decisions

Record important choices made during clarification and their rationale.

### Technical Approach

Describe the architecture and implementation strategy.

Explain why it fits the existing system and why it was selected over materially relevant alternatives.

### Affected Components

Identify the concrete modules, files, APIs, data models, or systems expected to change.

Do not invent file paths that have not been verified.

### Implementation Steps

Provide an ordered sequence of concrete implementation actions.

For each significant step, identify:

* what changes;
* where it changes;
* why it changes;
* important dependencies on earlier steps.

### Validation

Define how correctness will be established.

Include relevant:

* unit tests;
* integration tests;
* type or static checks;
* build checks;
* manual verification;
* regression checks;
* acceptance criteria.

A plan is incomplete if there is no credible way to verify that the requested behavior was achieved.

### Risks and Trade-offs

List meaningful implementation risks, compatibility concerns, migrations, or accepted compromises.

Do not pad this section with generic risks.

### Out of Scope

State important adjacent work that is intentionally excluded when the boundary could otherwise be misunderstood.

### Critical Files for Implementation

List the files or modules most central to the implementation, when they can be identified from repository evidence.

### Open Decisions

This section must contain:

`None.`

If it cannot truthfully contain `None.`, return to the clarification loop instead of finalizing the plan.

## Execution Boundary

This skill governs the planning phase.

While in Cursor Plan Mode:

* investigate;
* evaluate feasibility;
* ask clarification questions;
* critique the proposed design;
* produce and refine the implementation plan.

Do not implement code changes merely because the plan is complete.

Implementation begins only after the user explicitly transitions to execution, such as by approving or building the plan.

During implementation, do not introduce new material decisions silently.

If implementation reveals a previously unknown material ambiguity or invalidates a confirmed planning assumption, stop that decision path and ask the user using the same multiple-choice protocol before continuing.

## Prohibited Behavior

Never:

* guess a material user preference;
* silently choose between materially different behaviors;
* present speculation as a repository fact;
* assume that the user's proposed technical solution is correct;
* agree with a design merely to be cooperative;
* reject a design without a concrete technical reason;
* hide infeasibility behind a vague warning;
* produce a plan while knowingly leaving material decisions unresolved;
* silently weaken or reinterpret a requirement to make implementation easier;
* silently broaden the scope;
* silently add unnecessary abstractions or dependencies;
* claim that validation is possible without identifying a credible validation method;
* implement changes during the planning phase.

When uncertain whether a decision is material, ask whether different answers could change the resulting implementation or externally observable behavior. If they could, resolve it before finalizing the plan.

