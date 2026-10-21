---
name: factory-audit
description: Fourth factory station. Independently try to break a built change and write reproduced failures to factory/current/REVIEW.md with a PASS or REVISE verdict. Use in a new chat after build, or when the user says "audit", "review this", or "try to break it".
---

# Factory audit

Assume there is a bug and try to find it. You are the inspector at the end of the line: you did not watch the work being built, and you judge it only against the spec.

## Steps

1. If you built this change in this chat, tell the user to start a new chat and run `$factory-audit` there. Then stop. A reviewer who shares the builder's assumptions misses the same bugs.
2. Read `BRAIN.md`, then the goal, requirements, acceptance criteria, and edge cases in `factory/current/SPEC.md`. Do not read the plan or the build evidence yet: check the code, not the builder's story.
3. Find the change. Run `git status` and `git diff HEAD`, and read every new file. Ignore `factory/` and `audit-scratch/`. If there is no change, stop and say so. Never report "no findings" on an empty diff.
4. Write at least one attack for each angle:
   - **Edge inputs:** empty, very large, negative, duplicate, edges and corners, wrong type.
   - **State and timing:** do the action twice, undo it, do two things before the next step, do things in a different order.
   - **Missed requirement:** a criterion the code does not handle, even if the tests pass. Check the rules in `README.md` too.
   - **Old behavior:** something that worked before the change.
   - **Weak tests:** a test that would still pass if the code were wrong.
5. Try each attack in the terminal or with a test. Put any test you write in `audit-scratch/`.
6. Now read the build evidence in `SPEC.md`. Check that each PASS is backed by something real.
7. Delete `audit-scratch/`. Write `factory/current/REVIEW.md` with only the findings you reproduced, and show it to the user.

## Report (`factory/current/REVIEW.md`)

```markdown
# Audit, round 1

### F1: <one-line summary>
- Breaks: AC3
- Severity: high / medium / low
- Steps to reproduce:
  1. ...
- Expected: ...
- Actual: ...

## Attacks that found nothing
- ...

## Verdict
PASS, or REVISE (fix F1, F2)
```

## Rules

- A finding needs steps to reproduce. "This might be a problem" is not a finding.
- Do not fix anything.
- "No confirmed findings" with a PASS verdict is a valid result. Do not invent a bug.
- On PASS, the next station is `$factory-close`. On REVISE, it is `$factory-build` to fix the findings.
