---
name: adversarial-review
description: Try to break a finished change and write reproducible failures to REVIEW.md. Use after verify passes, or when the user says "adversarial review", "try to break this", or "red-team this change".
---

# Adversarial review

Assume there is a bug and try to find it. Judge the code only against the spec.

## Steps

1. If you wrote this code in this chat, tell the user to start a new chat and run this skill there. Then stop. A reviewer who shares the builder's assumptions misses the same bugs.
2. Read `SPEC.md`. If it is missing, stop and ask. Do not read the plan.
3. Find the change. Run `git status`, `git diff HEAD`, and `git log --oneline -5`. Read every new file, including files inside new folders. Ignore `REVIEW.md` and `review-scratch/`. If any commits are part of the feature, ask the user which commit it starts from and diff from there. If you find no change, stop and say so. Never report "no findings" on an empty diff.
4. Write at least one attack for each angle:
   - **Edge inputs:** empty, very large, negative, duplicate, special characters, wrong type.
   - **State:** refresh the page, go back, do the action twice, do two actions at once.
   - **Missed requirement:** a criterion the code does not handle, even if tests pass.
   - **Old behavior:** something that worked before the change.
   - **Weak tests:** a test that would still pass if the code were wrong.
5. Try each attack in the app, the terminal, or a test. Put any test you write in `review-scratch/` and do not commit it.
6. Delete `review-scratch/`. Keep only the findings you reproduced. Replace `REVIEW.md` with them and show the user.

## Report (`REVIEW.md`)

```markdown
### F1: <one-line summary>
- Breaks: AC3
- Severity: high / medium / low
- Steps to reproduce:
  1. ...
- Expected: ...
- Actual: ...

## Attacks that found nothing
- ...
```

## Rules

- A finding needs steps to reproduce. "This might be a problem" is not a finding.
- Do not fix anything.
- "No confirmed findings" is a valid result. Do not invent a bug.
