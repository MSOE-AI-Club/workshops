---
name: factory-shape
description: Second factory station. Turn factory/current/TICKET.md into SPEC.md with testable acceptance criteria and a checked plan, then wait for a person to approve it. Use after intake, or when the user says "shape", "write a spec", or "plan this".
---

# Factory shape

A bad spec is cheap to fix. Bad code is not. The spec gives the builder a target and gives the auditor something to check against.

## Steps

1. Read `BRAIN.md` and `factory/current/TICKET.md`. If the ticket is missing, or its verdict is not READY TO SHAPE, stop and tell the user to run `$factory-intake`.
2. Read the code the ticket touches.
3. Write the spec in `factory/current/SPEC.md` (format below): goal, requirements, acceptance criteria, edge cases, out of scope.
4. Write the plan: numbered steps. Each step names the files it changes and the acceptance criteria it covers.
5. Check your own plan:
   - Every acceptance criterion and edge case has a plan step. Fill in the coverage table. A MISSING row means you fix the plan.
   - List each assumption the plan makes about the code (file names, functions, data shapes). Open the file and mark the assumption TRUE or FALSE. Fix the plan for each FALSE.
   - Flag risky steps: steps that delete data, change many files, add a dependency, or change a shared interface.
6. Show the spec to the user and **wait for approval**. Change it if they ask. Set "Approved by the user" to yes only after they say so.

## Format (`factory/current/SPEC.md`)

```markdown
# <Feature name>

## Goal
One or two sentences: what the user can do when this is done.

## Requirements
- R1: The system must ...

## Acceptance criteria
- AC1 (R1): Given <state>, when <action>, then <result>.

## Edge cases
- E1: <case, for example an empty email>: <what must happen>

## Out of scope
- ...

## Plan
1. <change> in `<file>` (AC1, E2)

## Coverage
| Item | Plan step | How it is tested |
| --- | --- | --- |
| AC1 | 1 | test `test_...` |

## Assumptions checked
- "Wall collisions are checked in `Game.step()` in `snake/game.py`": TRUE

## Risks
- ...

Approved by the user: not yet
```

## Rules

- Each acceptance criterion must be something a person or a test can check. "Works well" is not a criterion. "When the head moves off the right edge, it appears on the left edge in the same row" is.
- Each requirement has at least one acceptance criterion.
- Include at least 3 edge cases (empty, duplicate, edges and corners, doing two things at once, very large, wrong type, ...).
- Do not write code in this station.
- When the user approves, tell them the next station is `$factory-build`.
