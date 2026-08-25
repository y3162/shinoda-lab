---
name: deep-research
description: >
  Conduct rigorous, multi-source research for questions where accuracy,
  evidence quality, novelty, comparison, or decision quality matters.
  Use for literature reviews, prior-art searches, novelty audits, technology
  surveys, method comparisons, hypothesis validation, market research,
  and other investigations requiring systematic web research.
icon: book-open
color: purple
---

# Deep Research

Conduct an auditable research investigation rather than a one-shot web search.

The goal is not to collect many links. The goal is to build defensible conclusions
whose supporting and contradicting evidence can be traced back to sources.

## When to use

Use this skill when the user asks for:

- deep research
- literature review
- prior-art search
- related-work investigation
- novelty assessment
- state-of-the-art survey
- systematic comparison
- hypothesis validation
- technology landscape
- evidence-based decision support
- comprehensive investigation of a difficult question

Do not use this workflow for simple factual questions that can be reliably answered
from one or two authoritative sources.

## Research depth

Choose one of the following unless the user specifies otherwise.

### shallow

Use for an orientation-level investigation.

- 2-3 research workstreams
- approximately 5-10 strong sources
- no extensive source archive required
- brief adversarial review

### standard

Default.

- 3-5 research workstreams
- approximately 10-30 useful sources
- source tracking
- explicit triangulation
- adversarial review
- citation verification

### deep

Use when novelty, publication, strategy, or a high-cost decision is involved.

- 5-8 research workstreams where useful
- continue searching until major claims stabilize
- prioritize primary and authoritative sources
- explicit counter-evidence search
- detailed source archive
- independent verification pass
- refresh targets for future updates

Do not treat source-count targets as quotas.
Stop when additional searches yield little new information and the important claims
are adequately supported or clearly marked as uncertain.

# Workflow

Follow the phases below in order.

## Phase 1: Reframe the question

Before searching, determine what the investigation is actually trying to establish.

Write:

1. the research question
2. the intended decision or output
3. scope
4. exclusions
5. relevant time range
6. 2-4 falsifiable hypotheses where appropriate

A good hypothesis must be capable of being contradicted by evidence.

For example:

- H1: Method A is the earliest published work that performs X.
- H2: Improvements attributed to X disappear under condition Y.
- H3: Existing methods operate only at utterance level, not frame level.

Do not convert descriptive questions into hypotheses when that would be unnatural.

## Phase 2: Create the research plan

Create:

`research/<slug>/plan.md`

The plan must contain:

- research question
- intended output
- scope
- exclusions
- hypotheses
- subquestions
- research workstreams
- preferred source types
- search strategy
- counter-evidence queries
- important ambiguities
- known risks
- stopping criteria

Break the investigation into independent workstreams whenever possible.

Examples:

- historical development
- strongest supporting evidence
- contradictory evidence
- recent work
- academic literature
- implementation evidence
- competing methods
- edge cases

Do not begin broad collection before the research question has been decomposed.

## Phase 3: Parallel reconnaissance

For standard and deep investigations, delegate independent research workstreams
to multiple `research-scout` subagents.

Run independent scouts in parallel whenever possible.

Do not run five scouts sequentially when they can be run concurrently.

Give each scout:

- the full research question
- the relevant hypothesis
- its specific workstream
- scope and time range
- requested source types
- expected output format

Different scouts should search meaningfully different evidence channels or questions.

Bad:

- Scout 1 searches "X method"
- Scout 2 searches "X method"
- Scout 3 searches "X method"

Good:

- Scout 1 searches academic literature
- Scout 2 traces historical prior art
- Scout 3 searches recent competing methods
- Scout 4 looks specifically for counterexamples
- Scout 5 investigates implementations or benchmark evidence

## Phase 4: Collect sources

Maintain:

`research/<slug>/sources.csv`

Recommended columns:

```text
id,title,authors,year,url,source_type,primary_or_secondary,relevance,credibility,accessed_at,notes
```

For important sources, create:

```text
research/<slug>/sources/S001_<slug>.md
research/<slug>/sources/S002_<slug>.md
...
```

Each source note should contain:

```md
# Source

## Metadata

- ID:
- Title:
- Authors / Organization:
- Date:
- URL:
- Source type:
- Primary / Secondary:

## Why it matters

## Relevant evidence

## Claims this source supports

## Claims this source contradicts

## Limitations

## Notes
```

Do not save large copied passages from sources.
Capture only the evidence needed to support analysis and preserve the source URL.

## Phase 5: Build atomic findings

For substantial investigations, store important findings separately:

```text
research/<slug>/findings/F01_<slug>.md
research/<slug>/findings/F02_<slug>.md
...
```

Each finding should represent one thesis.

Use:

```md
# Finding

## Claim

## Evidence

## Supporting sources

## Contradicting sources

## Confidence

high | medium | low

## Reasoning

## Remaining uncertainty
```

Do not combine several unrelated conclusions into one finding.

## Phase 6: Triangulate important claims

Do not treat search-result agreement as evidence.

For every central factual claim, determine:

1. how many independent sources support it
2. whether those sources are actually independent
3. whether different source types support it
4. whether primary evidence exists
5. whether contradictory evidence exists

As a default target, important claims should have:

- at least 3 independent sources when feasible
- preferably at least 2 different source types
- at least 1 primary or authoritative source when one exists

Examples of source types:

- peer-reviewed academic paper
- preprint
- official documentation
- standards body
- government publication
- company primary source
- dataset or benchmark
- reputable journalism
- technical implementation
- practitioner discussion

Three websites repeating the same press release count as one evidence lineage,
not three independent sources.

If a claim cannot be triangulated, do not hide that fact.

Label it as:

- weakly supported
- single-source
- disputed
- uncertain
- not verified

## Phase 7: Adversarial review

For standard and deep investigations, invoke `research-critic`.

The critic's job is not to summarize the research.

Its job is to try to make the current conclusions fail.

Ask it to look for:

- counterexamples
- contradictory papers
- older prior art
- missing baselines
- hidden assumptions
- dataset dependence
- benchmark artifacts
- selection bias
- temporal ambiguity
- source dependence
- alternative explanations
- overclaimed novelty
- unsupported causal claims

Save the result as:

`research/<slug>/adversarial-review.md`

After receiving the critique, perform additional searches for material objections.

Do not merely append the critique to the final report.
Update the conclusions when the critique changes the evidence.

## Phase 8: Verify evidence

Invoke `source-verifier` after the main investigation.

Give it:

- major claims
- source list
- findings
- adversarial review

The verifier should independently check:

- whether cited sources exist
- whether metadata is correct
- whether sources actually support the associated claim
- whether sources are independent
- whether important contradictory evidence was ignored
- whether primary sources were available but secondary sources were used instead
- whether dates are interpreted correctly
- whether novelty claims are sufficiently searched

Do not mark the research complete until major verification failures are resolved
or explicitly disclosed.

## Phase 9: Synthesize

Create:

`research/<slug>/YYYY-MM-DD_report.md`

Use this structure when appropriate:

```md
# Title

## Executive Summary

## Research Question

## Scope and Method

## Key Findings

## Evidence

## Counterevidence and Alternative Explanations

## Uncertainties and Limitations

## Conclusion

## Recommended Next Steps

## Sources
```

For novelty investigations, additionally include:

```md
## Novelty Assessment

### Closest Prior Work

### What Is Already Known

### What Appears Different

### Remaining Novelty Risk

### Search Areas That Could Still Invalidate Novelty
```

Separate clearly:

- observed evidence
- inference
- hypothesis
- recommendation

Never present inference as if it were directly stated by a source.

# Source policy

When source quality matters, read:

`references/source-policy.md`

Prefer evidence roughly in this order when appropriate:

1. original paper / primary research
2. official specification or documentation
3. authoritative institutional source
4. high-quality review
5. reputable secondary reporting
6. practitioner discussion
7. informal community discussion

This hierarchy is contextual, not absolute.

For questions about community experience, community sources may be the primary evidence.

# Search strategy

Use several query families rather than repeatedly paraphrasing one query.

## Discovery queries

Find terminology and major actors.

## Exact-concept queries

Search the precise mechanism or claim.

## Historical queries

Look for older terminology and predecessor methods.

## Citation-chain queries

Trace references backward and related work forward.

## Opposition queries

Search deliberately for evidence against the current hypothesis.

Examples:

```text
"<method>" limitation
"<claim>" contradiction
"<method>" failure
"<concept>" prior work
"<concept>" earlier method
"<concept>" benchmark
"<concept>" ablation
```

## Recent-work queries

Search explicitly for work after the strongest known source.

# Academic research rules

When researching academic literature:

- distinguish peer-reviewed papers from preprints
- record publication year
- record venue when known
- prefer the actual paper over summaries of the paper
- inspect Related Work and references when novelty matters
- search synonyms and older terminology
- do not assume an arXiv upload date is the first disclosure date
- distinguish publication date, conference date, and preprint date when relevant
- identify duplicated versions of the same work
- treat citation count as a discovery signal, not proof of correctness

For novelty claims, absence of found prior work is not proof of novelty.

Use language such as:

"Within the searched literature, I did not find..."

rather than:

"No prior work exists."

# Evidence discipline

Never fabricate:

- papers
- authors
- titles
- DOIs
- URLs
- benchmark numbers
- quotations
- publication venues

If a source cannot be opened or verified, say so.

If evidence is missing, reduce confidence instead of filling the gap from intuition.

When sources disagree, preserve the disagreement.

Do not force consensus.

# Temporal discipline

For time-sensitive questions:

- record the date of the search
- prefer recent primary evidence
- distinguish event date from article publication date
- check whether an apparently recent article is describing an older event

# Research completion criteria

Research is complete when:

- the important subquestions have been investigated
- central claims have adequate evidence or explicit uncertainty labels
- important counterarguments have been searched
- obvious source dependencies have been removed
- adversarial review has been addressed
- citations and metadata have been checked
- additional searches are producing diminishing informational returns

Research is not complete merely because many sources were collected.

# Refresh protocol

For standard/deep investigations that may become stale, create:

`research/<slug>/refresh_targets.md`

Include:

- claims likely to change
- relevant authors / organizations
- conferences or journals to revisit
- benchmark leaderboards
- standards or documentation pages
- suggested search queries
- last checked date

This should make future updates incremental rather than requiring the entire
investigation to be repeated.
