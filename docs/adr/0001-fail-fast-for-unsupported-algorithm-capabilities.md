# ADR 0001: Fail fast for unsupported algorithm capabilities

- Date: 2026-03-01
- Status: Accepted
- Issue: https://github.com/intellistream/sageVDB/issues/28

## Context

`VectorStore` had implicit fallback behavior in two places:

1. Unknown algorithm names were silently reassigned to `brute_force`.
2. When an algorithm did not support delete/update/range capability, operations silently marked index
   dirty and relied on later rebuild instead of failing immediately.

This hid capability boundaries and made runtime behavior ambiguous.

## Decision

1. Keep explicit `AUTO -> brute_force` selection only when requested via config intent.
2. Remove implicit reassignment for unknown algorithms; throw immediately.
3. For capability-mismatched operations on built index, throw immediately:
   - incremental add requires update capability
   - remove requires deletion capability
   - update requires both update and deletion capability
   - radius query requires range-search capability

## Consequences

- Unsupported operations fail fast with explicit errors.
- No hidden fallback path remains for capability mismatch.
- Runtime behavior is deterministic and aligned with algorithm capability contracts.

## Validation

- `pytest -q tests/test_issue28_capability_failfast.py`
- `ruff check tests/test_issue28_capability_failfast.py README.md`
