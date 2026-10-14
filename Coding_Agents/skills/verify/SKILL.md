---
name: verify
description: Check a finished change against each acceptance criterion and edge case in SPEC.md, with evidence. Use after implementing a spec, or when the user says "verify" or "check it against the spec".
---

# Verify the change

"The tests pass" is not the same as "the feature is correct". Check each criterion and show the evidence.

## Steps

1. Read `SPEC.md`. If it is missing, stop and ask the user what to check against. Do not invent criteria.
2. Run the project's checks (the test, lint, and build commands in the project README). Save the output.
3. For each acceptance criterion, find the evidence:
   - A test that checks it. Read the test and confirm it checks the right thing.
   - If no test exists, write one. Break the feature on purpose once and show the test fails. Undo the break by hand (never `git checkout` or `git restore`), then run the checks again and confirm they pass.
   - For UI criteria, do the action in the browser and take a screenshot or recording.
4. Try each edge case by hand at least once. Write down what you did and what happened.
5. Fill in the report.

## Report

```markdown
## Checks run
- `npm test`: 14 passed, 0 failed

## Criteria
| Item | Evidence | Result |
| --- | --- | --- |
| AC1 | test `adds item to list` | PASS |
| AC2 | screenshot of the error message | PASS |
| E1 | typed an empty name: app crashed | FAIL |

## Not checked
- Things you could not check, and why.
```

## Rules

- No PASS without evidence.
- Never change a test only to make it pass. If a test is wrong, say why.
- If something fails, report it. Do not fix it in this step.
