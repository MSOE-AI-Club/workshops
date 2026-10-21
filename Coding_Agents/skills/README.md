# Factory Skills

A **skill** is a folder with a `SKILL.md` file. The file tells a coding agent how to do one kind of task. The agent always sees the short `description` at the top. When your request matches it, or when you name the skill, the agent reads the full file and follows the steps.

These skills make a **software factory**: each skill is one station on an assembly line, and a shared memory file, the **factory brain** (`BRAIN.md`), makes the line better after every run. Four stations come ready-made. Students write the fifth, the audit, themselves during the workshop, from [`skill-template.md`](../starter/factory/skill-template.md). A reference answer for facilitators is in [`../solutions/factory-audit`](../solutions/factory-audit/SKILL.md); it is not in the zip. The idea comes from [Vercel's eve Software Factory](https://vercel.com/kb/guide/eve-software-factory) (stations: triage, plan, implement, independent review, plus a "factory brain") and from [Matt Pocock's skills](https://github.com/mattpocock/skills) (interview before you build, then test-first in small slices).

| Station | Skill | Reads | Writes | Person's job |
| --- | --- | --- | --- | --- |
| 1. Intake | [`factory-intake`](factory-intake/SKILL.md) | your request, `BRAIN.md` | `TICKET.md` | answer its questions |
| 2. Shape | [`factory-shape`](factory-shape/SKILL.md) | `TICKET.md` | `SPEC.md` (spec and plan) | **approve the spec** |
| 3. Build | [`factory-build`](factory-build/SKILL.md) | `SPEC.md` | code, tests, evidence | read the evidence |
| 4. Audit | `factory-audit`, written by the student (new chat) | `SPEC.md`, the diff | `REVIEW.md` ending in `Verdict: PASS` or `Verdict: REVISE` | write the skill, judge the findings |
| 5. Close | [`factory-close`](factory-close/SKILL.md) | everything | `BRAIN.md`, a commit | **approve the lessons** |

The run's files are in `factory/current/` in the project. Close moves them to `factory/done/`.

Why each station matters:
- **Intake** stops the factory from building the wrong thing, or something too big for one run.
- **Shape** turns a wish into acceptance criteria. A bad plan is cheap to fix. Bad code is not.
- **Build** works test-first, one criterion at a time, and shows evidence. Passing tests can still miss a requirement.
- **Audit** runs in a **new chat**, so the reviewer does not share the builder's assumptions. It sees the spec and the diff, not the builder's reasoning.
- **Close** asks what this run taught, and saves it to `BRAIN.md` so the next run starts smarter. A person approves every note, because every future run reads it.

## Install the skills

The workshop setup script copies these folders to `.agents/skills` in the workshop project (`%USERPROFILE%\agent-factory`). To use them in another project, copy the folders to the place your agent looks for skills:

- **Codex:** `.agents/skills/` in your project (or `~/.agents/skills/` for all projects)
- **Claude Code:** `.claude/skills/` in your project
- **Antigravity:** `.agent/skills/` in your project

The skills expect a `BRAIN.md`, an `AGENTS.md` that lists the project's check commands, and a `factory/` folder. Copy them from the [starter project](../starter).

## Use a skill

In Codex (terminal or VS Code), type `$` and pick the skill, or write it in your request: `$factory-intake Add a wrap-around mode for the walls`. Type `/skills` to list them. Start a new chat for the audit: `/new` in the terminal, or the **New chat** button in VS Code.

## Your tasks

1. **Read** each skill before you use it. Would a new teammate understand these steps?
2. **Run the line** on the workshop project, station by station. Write the audit station yourself when you get to it.
3. **Improve** one instruction. Pick a step where the agent did something wrong or vague. Save the agent's report, change the words, run the skill again on the same input, and compare the two reports.
4. **Add a station** of your choice. Copy `factory/skill-template.md` from the project to `.agents/skills/<your-skill-name>/SKILL.md`. The `name` must match the folder name and use only lowercase letters, numbers, and hyphens.

Ideas for a new station:
- `factory-grill`: before intake, interview the user one question at a time until the idea is clear (like Matt Pocock's `grill-me`)
- `factory-docs`: after build, update `README.md` for the change and test the steps in a clean folder
- `factory-commit-message`: write the commit message from the diff in your team's style
- `factory-retro`: read `factory/done/` and suggest which skill instruction to change, based on what the audits keep finding

## What makes a good skill

- The `description` says **what** the skill does and **when** to use it. Make it narrow, or it fires on requests it should not.
- The steps are short and in order.
- The skill says what to do when an input is missing.
- The skill says what "done" looks like and asks for **evidence** (a command and its output, a screenshot), not a claim like "it works".
- The skill says where a **person** must decide. Agents do the work; people make the judgment calls.
