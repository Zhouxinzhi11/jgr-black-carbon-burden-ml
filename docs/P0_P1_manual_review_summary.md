# P0/P1 manual review summary

The 405 code-tree `REVIEW_REQUIRED` records were prioritized without changing their authoritative code-audit classification.

| Priority | Count | Rc2 treatment |
|---|---:|---|
| P0 | 73 | Every item received an explicit resolved decision. Superseded scripts remain excluded; direct result/audit artifacts remain withheld where public aggregates provide the release substitute. |
| P1 | 298 | Auto-reviewed from import/call evidence, output names, manuscript links, and disclosure indicators; conservatively excluded from rc2. |
| P2 | 34 | Checkpoints, logs, opaque archives, and nonessential run artifacts; excluded from rc2. |

All P0 entries have a `RESOLVED_*` decision in `review_required_priority.csv`; **zero P0 items remain unresolved**. No file among the 405 was newly added to the public package. This release treatment does not retroactively reclassify the underlying audit rows.
