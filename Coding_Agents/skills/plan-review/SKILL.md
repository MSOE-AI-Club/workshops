---
name: plan-review
description: Check an implementation plan against SPEC.md before any code is written. Use after Plan Mode makes a plan, or when the user says "review the plan" or "is this plan good".
---

# Review a plan

Check the plan before the agent builds it.

## Steps

1. Read `SPEC.md`. Find the plan: in this chat, a plan file, or a plan artifact. If either one is missing, stop and ask the user for it. Do not invent a plan.
2. Make a table. Put each acceptance criterion and edge case in a row. Write which plan step covers it, or "MISSING".
3. List the assumptions the plan makes about the code (file names, functions, libraries, data shapes). Open the files and mark each one TRUE or FALSE.
4. Find risky steps: steps that delete data, change many files, add a dependency, or change a shared interface.
5. Check that the plan says how each row will be tested.

## Report

```markdown
## Coverage
| Criterion | Plan step | Status |
| --- | --- | --- |
| AC1 | Step 2 | Covered |
| E2 | - | MISSING |

## Assumptions checked
- "Scores are stored in `scores.json`": FALSE, they are in `db.sqlite`.

## Risks
- ...

## Verdict
APPROVE, or REVISE with the exact changes to make.
```

## Rules

- Do not write code during this review.
- Do not approve a plan with a MISSING row.
- Only list problems that change the plan.
