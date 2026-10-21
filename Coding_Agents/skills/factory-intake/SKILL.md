---
name: factory-intake
description: First factory station. Turn a raw request into factory/current/TICKET.md and decide if it is ready to shape. Use when the user brings a new feature request, bug report, or idea, or says "intake" or "new ticket".
---

# Factory intake

Decide what the request is, and if we should build it, before anyone plans or writes code.

## Steps

1. Read `BRAIN.md`.
2. If `factory/current/` already has files, a run is still open. Ask the user to close it (`$factory-close`) or to replace it. Stop until they answer.
3. Read the request. Look at the files it touches, only enough to size it. Do not plan the solution.
4. Classify it:
   - **Type:** feature, bug, or chore.
   - **Size:** S (one file), M (a few files), or L (many files or a new dependency). An L is too big for one run: propose how to split it into S or M tickets, then stop.
   - **For a bug:** try to reproduce it and write down the steps. If you cannot reproduce it, say so.
5. List the questions whose answers change what gets built. Ask the user at most 3. Write the other gaps down as assumptions.
6. Write `factory/current/TICKET.md` and show it to the user.

## Report (`factory/current/TICKET.md`)

```markdown
# <Short title>

- Type: feature / bug / chore
- Size: S / M / L
- Request: <the user's words>
- Areas: <files or modules it touches>
- Brain notes that apply: <from BRAIN.md, or "none">

## Questions for the user
1. ...

## Assumptions
- ...

## Verdict
READY TO SHAPE, NEEDS ANSWERS, or SPLIT (list the smaller tickets)
```

## Rules

- Do not write code, specs, or plans in this station.
- Do not mark a ticket READY TO SHAPE while a question is unanswered.
- When you are done, tell the user the next station is `$factory-shape`.
