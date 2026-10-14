---
name: write-spec
description: Turn a feature idea into SPEC.md with requirements, acceptance criteria, and edge cases. Use when the user says "write a spec" or "what are the requirements", or asks for a new feature that changes more than one file. Skip for small fixes.
---

# Write a spec

## Steps

1. Read the request. Look at the project files that the feature touches.
2. Ask the user only the questions that change what you build. Do not ask more than 3. For other gaps, write down your assumption.
3. If `SPEC.md` already exists, ask before you replace it.
4. Write the spec to `SPEC.md` in the format below.
5. Show the spec to the user and wait for approval before any planning or code.

## Spec format

```markdown
# <Feature name>

## Goal
One or two sentences: what the user can do when this is done.

## Requirements
- R1: The system must ...

## Acceptance criteria
- AC1 (R1): Given <state>, when <action>, then <result>.

## Edge cases
- E1: <case, e.g. empty input>: <what must happen>

## Out of scope
- Things we will not build now.

## Assumptions
- Things you decided without asking.
```

## Rules

- Each acceptance criterion must be something a person or a test can check. "Works well" is not a criterion. "Shows an error message under the field" is.
- Each requirement has at least one acceptance criterion.
- Include at least 3 edge cases (empty, very large, duplicate, wrong type, no network, ...).
