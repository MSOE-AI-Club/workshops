---
name: factory-build
description: Third factory station. Build the approved SPEC.md test-first and prove each acceptance criterion with evidence. Also fixes the findings in REVIEW.md. Use after the spec is approved, or when the user says "build it", "implement the spec", or "fix the findings".
---

# Factory build

"The tests pass" is not the same as "the feature is correct". Build one criterion at a time, and show evidence for each one.

## Steps

1. Read `BRAIN.md` and `factory/current/SPEC.md`. If the spec is missing or not approved by the user, stop and say so.
2. If `factory/current/REVIEW.md` exists with verdict REVISE, you are fixing findings. Go to "Fix findings" below.
3. Run the tests before you change anything, so you know the starting point.
4. For each acceptance criterion, in plan order, do one small loop:
   1. Write one test for that criterion. Run it and see it fail for the right reason.
   2. Write the smallest code that makes it pass.
   3. Run all the tests.

   Do not write all the tests first and all the code after. Each test should come from what you learned in the loop before it.
5. Try each edge case by hand at least once (in the terminal or with a test). Write down what you did and what happened.
6. Run all the checks in `AGENTS.md`. Add the report to the bottom of `SPEC.md`.
7. Tell the user the next station is `$factory-audit`, in a **new chat**.

## Fix findings

Take the findings in `REVIEW.md` in severity order, one at a time:

1. **Reproduce** it. If it does not fail, mark it NOT REPRODUCED and do not change code.
2. **Write a test** that shows the bug. See it fail.
3. **Fix** the smallest amount of code that makes the test pass. Run all the tests.

Then update the build evidence and send the work back to `$factory-audit` in a new chat. If two audit rounds still say REVISE, stop and ask the user what to do.

## Report (bottom of `SPEC.md`)

```markdown
## Build evidence
- `uv run pytest`: 14 passed, 0 failed

| Item | Evidence | Result |
| --- | --- | --- |
| AC1 | test `test_wrap_right_edge_comes_back_on_left` | PASS |
| E1 | test `test_wrap_into_own_body_still_ends_game`: game over after the step | PASS |

## Not checked
- <what you could not check, and why>
```

## Rules

- No PASS without evidence.
- Do not change the spec. If the spec is wrong, stop and tell the user.
- Never delete or weaken a test to make it pass.
- Do not change code that the spec does not need.
