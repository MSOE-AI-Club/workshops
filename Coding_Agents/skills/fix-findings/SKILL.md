---
name: fix-findings
description: Reproduce and fix the findings in REVIEW.md, prove each fix, then verify again. Use after adversarial-review, or when the user says "fix the findings" or "fix the review".
---

# Fix findings

## Steps

1. Read `REVIEW.md`. If it is missing, ask the user to paste the findings.
2. Take the findings in severity order, one at a time:
   1. **Reproduce.** Follow the steps. If it does not fail, mark it NOT REPRODUCED, tell the user, and do not fix it.
   2. **Write a failing test** that shows the bug. Run it and confirm it fails. If the bug cannot be tested automatically (for example a page refresh), say so and use the reproduction steps as the check.
   3. **Fix** the smallest amount of code that makes the check pass.
   4. **Run all tests.** The new test passes and no old test fails.
3. When all findings are done, use the `verify` skill again.

## Report

```markdown
| Finding | Reproduced? | Proof of fix | Fixed? |
| --- | --- | --- | --- |
| F1 | Yes | test `rejects empty name` | Yes |
| F2 | No | - | - |
```

## Rules

- Do not change other code "while you are there". Keep each fix small.
- Do not delete or weaken a test to make it pass.
