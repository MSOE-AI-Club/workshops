# Workshop Skills

A **skill** is a folder with a `SKILL.md` file. The file tells a coding agent how to do one kind of task. The agent always sees the short `description` at the top. When your request matches it, the agent reads the full file and follows the steps.

These skills follow the workshop workflow. They pass work to each other through files in your project:

| Step | Skill | Reads | Writes |
| --- | --- | --- | --- |
| 1. Requirements | [`write-spec`](write-spec/SKILL.md) | your request | `SPEC.md` |
| 2. Plan | Plan Mode (built in) | `SPEC.md` | a plan |
| 3. Check the plan | [`plan-review`](plan-review/SKILL.md) | `SPEC.md`, the plan | a coverage table |
| 4. Build | the agent, no skill | `SPEC.md`, the plan | code and tests |
| 5. Check the work | [`verify`](verify/SKILL.md) | `SPEC.md`, the code | evidence for each criterion |
| 6. Try to break it | [`adversarial-review`](adversarial-review/SKILL.md) | `SPEC.md`, the diff | `REVIEW.md` |
| 7. Fix | [`fix-findings`](fix-findings/SKILL.md) | `REVIEW.md` | fixes and tests |

Why each step matters:
- A clear spec gives the agent a target, and gives you something to check its work against.
- A bad plan is cheap to fix. Bad code is not.
- Passing tests can still miss a requirement. Ask for evidence.
- The agent that wrote the code tends to trust it. Run step 6 in a **new chat** so the reviewer does not share the builder's assumptions.

## Install the skills

Copy the skill folders into the place your agent looks for skills:

- **Antigravity:** `.agent/skills/` in your project (or `~/.gemini/antigravity/skills/` for all projects)
- **Claude Code:** `.claude/skills/` in your project

## Use a skill

Name the skill in your request ("use the verify skill"), or describe the task with words from its description ("review the plan"). In Claude Code you can also type `/verify`.

## Your tasks

1. **Read** each skill before you use it. Would a new teammate understand these steps?
2. **Use** them on the workshop project, in order.
3. **Improve** one instruction. Pick a step where the agent did something wrong or vague. Save the agent's report, change the words, run the skill again on the same input, and compare the two reports.
4. **Make** a new skill of your choice. Copy [`../skill-template.md`](../skill-template.md) to `<your-skill-name>/SKILL.md`. The `name` must match the folder name and use only lowercase letters, numbers, and hyphens.

Ideas for your skill:
- `write-readme`: write setup steps for a project and test them in a clean folder
- `commit-message`: write a commit message from the diff in your team's style
- `explain-code`: explain a file to a beginner with one small example
- `ui-check`: open each page at phone and desktop width and screenshot it
- `dependency-check`: check a new library before you add it (license, size, last update)

## What makes a good skill

- The `description` says **what** the skill does and **when** to use it. Make it narrow, or it fires on requests it should not.
- The steps are short and in order.
- The skill says what to do when an input is missing.
- The skill says what "done" looks like and asks for **evidence** (a command and its output, a screenshot), not a claim like "it works".
