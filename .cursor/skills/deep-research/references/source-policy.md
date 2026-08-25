# Source Quality Policy

Use source quality as part of the reasoning process rather than treating all URLs
as equally reliable.

## Source classes

### A. Primary authoritative evidence

Examples:

- original research paper
- official specification
- official documentation
- government data
- standards body
- original dataset
- benchmark results from maintainers
- original company announcement

Preferred for factual claims about what a method, system, policy, or organization does.

### B. Strong secondary evidence

Examples:

- systematic review
- high-quality survey paper
- reputable technical review
- major investigative journalism
- respected institutional analysis

Useful for synthesis, context, and discovering primary sources.

### C. Contextual evidence

Examples:

- technical blogs
- conference presentations
- expert commentary
- implementation writeups
- GitHub repositories

Useful for implementation details and practical behavior.

### D. Community evidence

Examples:

- forums
- Reddit
- issue trackers
- social media

Useful for:

- user experience
- failure modes
- emerging problems
- practical adoption

Do not use community popularity as proof of a technical claim.

# Independence

Sources are not independent merely because they have different URLs.

Check whether they:

- cite the same original source
- repeat the same press release
- belong to the same organization
- reproduce the same dataset
- summarize the same paper
- copy one another

Trace important claims back to their evidence origin.

# Primary-source preference

When a secondary source makes an important factual claim:

1. identify its cited primary source
2. open the primary source when accessible
3. verify the relevant claim there
4. cite the primary source preferentially

Secondary sources may still be useful for interpretation.

# Academic evidence

For papers record when available:

- title
- authors
- year
- venue
- DOI
- arXiv identifier
- peer-reviewed / preprint status
- dataset
- experimental conditions

Do not compare reported numbers without checking whether evaluation conditions differ.

# Novelty evidence

A novelty claim requires broader searching than an ordinary factual claim.

Search:

- exact terminology
- synonyms
- older terminology
- neighboring fields
- references of closest papers
- papers citing closest work
- patents where relevant
- workshop and preprint literature where relevant

Classify novelty confidence:

## High

Extensive searches across terminology and citation chains found no equivalent method,
and the closest prior work has identifiable differences.

## Medium

No equivalent work was found, but search coverage has meaningful gaps.

## Low

Potentially similar prior work exists or terminology makes exhaustive search difficult.

Never equate "not found" with "does not exist."

# Contradictory evidence

Contradictory sources must not be silently discarded.

Record:

- what conflicts
- whether conditions differ
- which source is stronger
- whether both results can be true under different settings

When disagreement cannot be resolved, report the disagreement explicitly.

# Confidence

Use confidence labels only after considering:

- evidence quantity
- evidence independence
- source quality
- directness
- consistency
- recency
- unresolved counterevidence

Suggested labels:

- high
- medium
- low

Confidence describes the evidence available for the claim, not subjective certainty.
