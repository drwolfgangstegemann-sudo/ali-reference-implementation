# Architectural findings exposed by the executable implementation

The implementation revealed several points that were clarified or corrected
between v0.2 and v0.3.

## Resolved in v0.3

### 1. SuperEgo must not hold a reference to EnvironmentInterface

In v0.2 the SuperEgo received the full EnvironmentInterface in order to call
`is_inside_domain()`. This coupled normative evaluation to execution concerns.

**Fix:** The SuperEgo now receives only `workspace: Path` and implements
`_is_inside_domain()` internally using path resolution. The EnvironmentInterface
is never passed to the SuperEgo. Test: `test_super_ego_has_no_environment_attribute`.

### 2. target=None must not trigger H2_LOCAL_DOMAIN

In v0.2 a proposal with `target=None` was rejected as an H2 violation. This
incorrectly blocks action types that do not require a destination path.

**Fix:** The H2 check for the target is now guarded by `if proposal.target is not None`.
Test: `test_target_none_does_not_trigger_h2`.

### 3. A3_MINIMUM_VIABILITY was inconsistently categorised

In v0.2 the viability gate was labelled with an "A" (adaptive) prefix but behaved
as a permanent constraint. Since the execution threshold is fixed in configuration
and cannot change at runtime, it belongs to the hard norm category.

**Fix:** The norm is renamed `H6_MINIMUM_VIABILITY` and added to `HARD_NORMS`.
Test: `test_low_viability_rejects_via_h6`.

### 4. _sanitise_name could produce an empty stem

In v0.2 a filename whose stem consisted entirely of characters outside the
allowed set would produce a bare extension such as `.txt`.

**Fix:** An explicit fallback sets the stem to `"unnamed"` when sanitisation
leaves it empty. Tests: `test_sanitise_name_empty_stem_falls_back_to_unnamed`.

### 5. Ego ignored the history parameter

In v0.2 `generate()` accepted `history: Sequence[Event]` but never used it.

**Fix:** The Ego now uses file-specific history in two ways:
- If any past Event for this file shows `execution.success == True`, the Ego
  returns no proposals (the file has already been handled).
- If the same rule was previously rejected for this file, confidence is reduced
  by 0.05 per rejection, reflecting that the proposal has already failed
  normative evaluation in this context.

Global learning (`successful_rules`) now also contributes a small confidence
boost (+0.001 per successful execution of the rule across all files).
Tests: `test_ego_skips_file_already_successfully_processed`,
`test_ego_history_lowers_confidence_after_rejection`.

### 6. Rejected proposals were silently discarded by the break statement

In v0.2 the inner proposal loop exited after the first proposal regardless
of its normative decision. Rejected proposals were stored but no further
proposals in the list were evaluated.

**Fix:** The `break` now fires only after an *approved* proposal. Rejected
proposals are stored as Events and evaluation continues with the next proposal
in the list. This implements the iterative Ego / Super-Ego interaction described
in the specification.
Test: `test_rejected_proposals_are_stored_as_events`.

### 7. Super-Ego adaptive learning was verified by test

The `learn()` method in SuperEgo increments adaptive weights after approved
cycles. This was not tested in v0.2.
Test: `test_super_ego_adaptive_weights_increase_after_success`.

---

## Remaining open items (not addressed in v0.3)

### Adaptive learning state is ephemeral

`Ego.successful_rules` and `SuperEgo.adaptive_weights` live only in process
memory. Every restart resets them. A production implementation requires
persistent normative state (e.g. a `normative_state` table in SQLite).

### Causal Core learning is a counter only

`CausalCore.learn()` increments `completed_events` but the counter has no
effect on viability scoring. A real deployment needs a domain-specific
viability model with evidence-based parameter adaptation.

### Cycle vs. Event distinction

The specification sometimes implies one Event per observation cycle. The
implementation creates one Event per evaluated proposal, which may be several
per observation. A future version may introduce a parent `Cycle` object
grouping all Events for one observation.

### One proposal per observation when first is rejected

When all proposals for an observation are rejected, the Runtime cannot currently
request additional proposals because the Ego generates its full list at once.
A future version may implement true iterative regeneration.
