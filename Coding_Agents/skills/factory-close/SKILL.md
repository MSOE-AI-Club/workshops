---
name: factory-close
description: Last factory station. Close a finished run, ask the user which lessons to save in BRAIN.md, archive the run, and commit. Use after an audit verdict of PASS, or when the user says "close", "ship it", or "wrap up".
---

# Factory close

The factory only gets smarter if each run leaves a lesson for the next one. That memory is `BRAIN.md`, and every future run reads it.

## Steps

1. Read everything in `factory/current/`. If `REVIEW.md` is missing or its verdict is not PASS, tell the user and ask if they want to close anyway. Stop until they answer.
2. Run all the checks in `AGENTS.md` one more time. If one fails, stop and send the work back to `$factory-build`.
3. Propose 1 to 3 lessons for `BRAIN.md`. A good lesson is something a future run would get wrong without it:
   - a build or test quirk you hit,
   - a gotcha in this code,
   - a finding the audit caught that the builder should have caught.

   A summary of the feature is not a lesson. Neither is advice that is true for every project.
4. Show the lessons and **wait for the user** to approve, edit, or drop each one. Add only the approved ones, to the right section of `BRAIN.md`. If a lesson repeats a note that is already there, improve that note instead of adding a new one.
5. Move `factory/current/` to `factory/done/<NNN>-<short-name>/`, where NNN is the next number (001, 002, ...).
6. Commit all the changes. First line: the ticket title. Then one line per acceptance criterion. If the agent asks for permission to run git, that is expected.

## Report

```markdown
Closed: <ticket title>
- Audit rounds: <1 or 2>
- Lessons added to BRAIN.md: <list, or none>
- Commit: <hash> <first line of the message>
```

## Rules

- Never write to `BRAIN.md` without the user's approval. Every future run reads it, so one wrong note misleads all of them.
- Do not change code in this station.
