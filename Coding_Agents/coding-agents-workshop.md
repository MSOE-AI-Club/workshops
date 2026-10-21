Coding Agents: Build a Software Factory
===
MSOE AI Club Workshop

```
  _____________
 /0   /     \  \
/  \ M A I C/  /\
\ / *      /  / /
 \___\____/  @ /
          \_/_/
```
(Rosie is not needed!)

Prereqs:
- Download `coding-agents-workshop.zip` from the AI Club Teams post and unzip it.
- Run the setup script from the `setup` folder ([full steps](setup/README.md)). It installs everything and signs you in with the workshop key. No OpenAI account needed.
  - Windows (PowerShell): `powershell -ExecutionPolicy Bypass -File setup.ps1`
  - Mac (Terminal): `bash setup.sh`
- At the workshop, run it again with `-Check` (Windows) or `--check` (Mac). Every required row should say `[OK]`.

Most of this guide is reading. When you hit a ⚠️ **STOP** box, copy its prompt into the Codex chat and send it.

---

```
 _    ___ ___ ___  ___  _  _   _
| |  | __/ __/ __|/ _ \| \| | / |
| |__| _|\__ \__ \ (_) | .` | | |
|____|___|___/___/\___/|_|\_| |_|
```

## Models, harnesses, and agents

A **model** takes text in and gives text back. On its own it can't see your files, can't run code, and remembers nothing. One question to a bare model gets you a guess.

A **harness** wraps the model and adds three things:

1. **Context:** your request, your code, the error, the project's instructions.
2. **A conversation:** follow-ups build on each other. (The whole chat is re-sent every time.)
3. **The ability to run things on its own:** the model decides to run Python, `ls`, your tests, or to ask *you* a question. The harness runs it and hands the result back.

ChatGPT is a harness. **Codex is a harness built for code.** A **coding agent** is a model in a harness, looping on its own:

```
 Inside the harness (Codex):

 ┌──► 1. Gather context: your request, AGENTS.md, skills,
 │       the files opened so far, everything said so far
 │                         │
 │                         ▼
 │                  ┌─────────────┐  reads all of it, then either
 │                  │    MODEL    │  runs a tool or says it's done
 │                  └──────┬──────┘
 │                         │
 │            ┌────────────┴────────────┐
 │            ▼                         ▼
 │       2. Run the tool:           It's done.
 │          read or edit a file,        │
 │          run a command, or           │
 │          ask you a question          │
 │            │                         │
 │            ▼                         │
 └─── 3. Add the result to the          │
        context, then go to step 1      ▼
                            4. Return the answer to you
```

**Q: If it can run the tests, how does it hand you broken code?**

**A: It only checks what it knows to check,** and it only knows what's in its **context window**. A new chat starts empty.

Other coding harnesses: Claude Code, Cursor, Google Antigravity, GitHub Copilot. Today Codex runs a small, cheap model: `gpt-6-luna`.

```
 _    ___ ___ ___  ___  _  _   ___
| |  | __/ __/ __|/ _ \| \| | |_  )
| |__| _|\__ \__ \ (_) | .` |  / /
|____|___|___/___/\___/|_|\_| /___|
```

## AGENTS.md: what the agent should always know

Every new chat starts empty, so you end up repeating yourself: which tools to use, how to run the tests, and rules no model could guess ("prices are in cents," "never touch `migrations/`").

**`AGENTS.md`** is a file at the top of your project that the harness loads into **every** chat. Write it once. Ours, trimmed:

```markdown
# Agent instructions
This is the classic Snake game in Python, drawn with pygame.
The rules it must follow are in `README.md`.

## Commands
- Run the tests: `uv run pytest`
- Play the game: `uv run python -m snake`

## Rules
- Keep game rules in `snake/game.py`, with no pygame in it.
- You can't see the game window, so test the rules instead of playing.
- Do not add a dependency without asking.
```

Tips: keep it short (it's in every chat), be specific, and add to it whenever you correct the agent twice. (Claude Code reads `CLAUDE.md` instead.)

```
 _    ___ ___ ___  ___  _  _   ____
| |  | __/ __/ __|/ _ \| \| | |__ /
| |__| _|\__ \__ \ (_) | .` |  |_ \
|____|___|___/___/\___/|_|\_| |___/
```

## Skills: how to do one particular job

`AGENTS.md` is for what's always true. A **skill** is a recipe for one job you repeat, like "review this change." It's a folder with a `SKILL.md`:

```markdown
---
name: factory-intake
description: Turn a raw request into factory/current/TICKET.md. Use when
  the user brings a new feature request, bug report, or idea.
---
# Factory intake
## Steps
1. Read `BRAIN.md`.
2. Read the request and the files it touches.
3. Ask the user at most 3 questions.
4. Write `factory/current/TICKET.md`.
```

At the start of a chat, the harness loads only each skill's `name` and `description`. When you type `$factory-intake` (or ask for something matching the description), it loads the rest.

Why skills: you stop repeating yourself, the job gets done the same way every time, and fixing the skill once fixes every future run.

**A prompt is for this task. `AGENTS.md` is for every task. A skill is for a kind of task.**

(Codex: `.agents/skills/`. Claude Code: `.claude/skills/`. Same format.)

```
 _    ___ ___ ___  ___  _  _   _ _
| |  | __/ __/ __|/ _ \| \| | | | |
| |__| _|\__ \__ \ (_) | .` | |_  _|
|____|___|___/___/\___/|_|\_|   |_|
```

## Why one chat isn't enough

Getting an agent to write code is easy. Getting the *right* code, every time, is what matters for real work. When agents fail, it's rarely the model. It's the **process**. Ask one chat to "add wrap-around walls to Snake" and:

1. **It guesses what you meant.** Nobody wrote down what "done" looks like.
2. **It grades its own homework.** The same agent writes the code and the tests.
3. **It forgets.** Next week's chat makes the same mistake.

## The software factory

Split the work into **stations**, each a skill with one job. A person makes the calls between them, and a shared **factory brain** (`BRAIN.md`) remembers lessons. Four skills come ready-made; you'll write the fifth.

```
 request
    │
    ▼
 1. INTAKE ── What is it? Should we build it?           → TICKET.md
    │
    ▼
 2. SHAPE ─── What exactly is "done"?   (YOU APPROVE)   → SPEC.md
    │
    ▼
 3. BUILD ─── one test, then the code, then evidence    → code + tests
    │    ▲
    ▼    │ REVISE: fix the findings
 4. AUDIT ─── New chat. Try to break it. (YOU JUDGE)    → REVIEW.md
    │
    ▼
 5. CLOSE ─── What did we learn?        (YOU APPROVE)   → BRAIN.md + commit

 BRAIN.md: the factory brain. Every station reads it first.
```

| Station | In team terms | Fixes |
| --- | --- | --- |
| Intake | the **ticket**: requirements, the *what* | guessing |
| Shape | the **technical spec**: the *how*, and how we'll know it works | guessing |
| Build | the **implementation** | |
| Audit | **code review and QA**, by someone who didn't write it | grading its own homework |
| Close | the **retro** | forgetting |

The agents do the work. **You make the calls.** None of this needs a smarter model.

Borrowed from Vercel's [eve Software Factory](https://vercel.com/kb/guide/eve-software-factory) and Matt Pocock's [skills](https://github.com/mattpocock/skills).

---

```
 _    ___ ___ ___  ___  _  _   ___
| |  | __/ __/ __|/ _ \| \| | | __|
| |__| _|\__ \__ \ (_) | .` | |__ \
|____|___|___/___/\___/|_|\_| |___/
```

## Let's build something

Setup put the **bones of a Snake game** in `agent-factory` in your user folder: a window with an empty board, and TODOs. Your factory builds the rest. Open it:

```text
code ~/agent-factory
```

Click the **Codex** icon in the side bar (or Ctrl/Cmd+Shift+P, **Codex: Open Codex Sidebar**). Run `uv run python -m snake` in the VS Code terminal to see the empty board (Esc quits).

What's in the project:

- `README.md`: the game's rules. `snake/game.py`: the rules in code (TODO). `snake/app.py`: the window (TODO).
- `requirements.txt`: the packages (`pygame-ce`, `pytest`), already installed in `.venv`.
- `AGENTS.md`, `BRAIN.md`: always-on instructions and the factory brain.
- `.agents/skills/`: four stations. Audit is missing; you'll write it.
- `factory/`: each run's paperwork, plus `BACKLOG.md` (game ideas) and `skill-template.md`.

> ```
>  ____ _____ ___  ____    _
> / ___|_   _/ _ \|  _ \  / |
> \___ \ | || | | | |_) | | |
>  ___) || || |_| |  __/  | |
> |____/ |_| \___/|_|     |_|
> ```
>
> ### ⚠️ Say hi to your agent
>
> ```text
> What does this project do? Run the tests, then list the factory skills in .agents/skills and tell me in one line what each one does.
> ```
>
> **What happens:** Codex reads the project, runs the tests (1 passes; there's no game yet), and opens the skills. Each command it runs is one trip around the harness loop.
>
> **Check:** type `$` in the chat box. Four `factory-` skills should appear.

```
 ___ _  _ _____ _   _  _____
|_ _| \| |_   _/_\ | |/ / __|
 | || .` | | |/ _ \| ' <| _|
|___|_|\_| |_/_/ \_\_|\_\___|
```

### Station 1: Intake (the ticket)

What's being asked for and why, in plain language. No *how* yet. Intake asks at most 3 questions and writes down the rest as assumptions.

> ```
>  ____ _____ ___  ____    ____
> / ___|_   _/ _ \|  _ \  |___ \
> \___ \ | || | | | |_) |   __) |
>  ___) || || |_| |  __/   / __/
> |____/ |_| \___/|_|     |_____|
> ```
>
> ### ⚠️ Hand intake the ticket
>
> ```text
> $factory-intake Build the Snake game. The rules are in README.md. Fill in the TODOs in snake/game.py and snake/app.py.
> ```
>
> **What happens:** Codex asks up to 3 questions, then writes `factory/current/TICKET.md`.
>
> **Check:** answer however you like (you're the product owner). `TICKET.md` should end with **READY TO SHAPE**.

```
 ___ _  _   _   ___ ___
/ __| || | /_\ | _ \ __|
\__ \ __ |/ _ \|  _/ _|
|___/_||_/_/ \_\_| |___|
```

### Station 2: Shape (the technical spec)

*How* we'll build it, and exactly how we'll know it works. The key part is **acceptance criteria**: sentences a test can check.

- Bad: "Eating works correctly."
- Good: "When the head moves onto the food, the snake grows by one square and the score goes up by 1."

Shape waits for your approval. Nothing gets built until you say so.

> ```
>  ____ _____ ___  ____    _____
> / ___|_   _/ _ \|  _ \  |___ /
> \___ \ | || | | | |_) |   |_ \
>  ___) || || |_| |  __/   ___) |
> |____/ |_| \___/|_|     |____/
> ```
>
> ### ⚠️ Shape the spec, then approve it
>
> ```text
> $factory-shape
> ```
>
> **What happens:** Codex writes `factory/current/SPEC.md` with criteria, edge cases, and a plan, then waits.
>
> **Check:** can a test check every criterion? Any **MISSING** rows? Does it follow *every* rule in `README.md`? Ask for changes, then send:
>
> ```text
> Approved.
> ```

```
 ___ _   _ ___ _    ___
| _ ) | | |_ _| |  |   \
| _ \ |_| || || |__| |) |
|___/\___/|___|____|___/
```

### Station 3: Build (the implementation)

One criterion at a time, test first: write a test, watch it **fail**, write just enough code to pass, repeat. Then show evidence for every criterion. "Trust me" doesn't count.

Codex edits files in the project on its own, but asks before anything outside it. Read the command before you approve.

> ```
>  ____ _____ ___  ____    _  _
> / ___|_   _/ _ \|  _ \  | || |
> \___ \ | || | | | |_) | | || |_
>  ___) || || |_| |  __/  |__   _|
> |____/ |_| \___/|_|        |_|
> ```
>
> ### ⚠️ Build it
>
> ```text
> $factory-build
> ```
>
> **What happens:** red, green, red, green through the criteria (a few minutes), then a **Build evidence** table at the bottom of `SPEC.md`.
>
> **Check:** play it (arrows steer, R restarts, Esc quits). Does it follow every rule in `README.md`?
>
> ```text
> uv run python -m snake
> ```

> 💡 **Done early?** Did any new test pass the first time it ran? A test that never failed might not test anything.

```
   _  _   _ ___ ___ _____
  /_\| | | |   \_ _|_   _|
 / _ \ |_| | |) | |  | |
/_/ \_\___/|___/___| |_|
```

### Station 4: Audit (code review and QA)

An inspector who didn't watch the build. It runs in a **new chat**, so it doesn't share the builder's blind spots. It assumes there's a bug, and a finding only counts with steps to reproduce.

**You write this station.** The one hard requirement: it writes `factory/current/REVIEW.md` ending in `Verdict: PASS` or `Verdict: REVISE`, because build and close look for that.

> ```
>  ____ _____ ___  ____    ____
> / ___|_   _/ _ \|  _ \  | ___|
> \___ \ | || | | | |_) | |___ \
>  ___) || || |_| |  __/   ___) |
> |____/ |_| \___/|_|     |____/
> ```
>
> ### ⚠️ Write your audit skill
>
> ```text
> Help me write a new skill: the fourth factory station, an auditor. Use factory/skill-template.md as the format and save it as .agents/skills/factory-audit/SKILL.md.
>
> The auditor reviews a finished change like a strict code reviewer who didn't watch it being built. It should:
> - refuse to run if it wrote the code in this same chat, and tell me to start a new chat
> - read BRAIN.md, the requirements and acceptance criteria in factory/current/SPEC.md, and the rules in README.md
> - find the change with git status and git diff
> - try hard to break it, and only report problems it can reproduce with exact steps
> - never fix any code
> - write factory/current/REVIEW.md with steps to reproduce, expected, and actual for each finding, and end with "Verdict: PASS" or "Verdict: REVISE"
>
> Before you write it, ask me what I think a good reviewer should always check. Show me the skill before you save it.
> ```
>
> **What happens:** Codex asks what a good reviewer checks, then shows you a draft `SKILL.md`.
>
> **Check:** does the `description` say *what* and *when*? Would two agents follow the steps the same way? Edit, then tell it to save.

> ```
>  ____ _____ ___  ____     __
> / ___|_   _/ _ \|  _ \   / /_
> \___ \ | || | | | |_) | | '_ \
>  ___) || || |_| |  __/  | (_) |
> |____/ |_| \___/|_|      \___/
> ```
>
> ### ⚠️ Run your audit in a new chat
>
> Start a **new chat** (the **+** at the top of the Codex panel), then:
>
> ```text
> $factory-audit
> ```
>
> **What happens:** your skill writes `factory/current/REVIEW.md` with a verdict. (Skill not showing up? Check the path, then open another new chat.)
>
> **Check:** reproduce one finding yourself. **PASS:** go to Station 5. **REVISE:** in the build chat, send the prompt below, then audit again in a new chat. Two rounds max.
>
> ```text
> $factory-build Fix the findings in factory/current/REVIEW.md.
> ```

```
  ___ _    ___  ___ ___
 / __| |  / _ \/ __| __|
| (__| |_| (_) \__ \ _|
 \___|____\___/|___/___|
```

### Station 5: Close (the retro)

Close proposes 1 to 3 **lessons** for `BRAIN.md`: things a future run would get wrong without them. You approve each one, because every future run reads that file.

> ```
>  ____ _____ ___  ____    _____
> / ___|_   _/ _ \|  _ \  |___  |
> \___ \ | || | | | |_) |    / /
>  ___) || || |_| |  __/    / /
> |____/ |_| \___/|_|      /_/
> ```
>
> ### ⚠️ Close the run
>
> ```text
> $factory-close
> ```
>
> **What happens:** Codex checks the verdict, reruns the tests, proposes lessons, and waits. Then it updates `BRAIN.md`, moves the run to `factory/done/`, and commits.
>
> **Check:** approve, edit, or drop each lesson. Look at the new `BRAIN.md` and the commit.

**Q: Did your audit find the classic Snake bug?** Lots of Snake games, human- or agent-written, have one that the tests usually miss. Hint: reread the `README.md` rule about pressing keys fast, then try it.

> 💡 **Done early?** Run another idea from `factory/BACKLOG.md` (wrap-around walls, high score, two players, Tron...) through all five stations.

---

```
 _    ___ ___ ___  ___  _  _    __
| |  | __/ __/ __|/ _ \| \| |  / /
| |__| _|\__ \__ \ (_) | .` | / _ \
|____|___|___/___/\___/|_|\_| \___/
```

## Make the factory yours

When a station gets something wrong, don't fix the code. **Fix the instructions,** and every future run improves. Pick one.

### Option A: Improve a station

> ```
>  ____ _____ ___  ____     ___    _
> / ___|_   _/ _ \|  _ \   ( _ )  / \
> \___ \ | || | | | |_) |  / _ \ / _ \
>  ___) || || |_| |  __/  | (_) / ___ \
> |____/ |_| \___/|_|      \___/_/   \_\
> ```
>
> ### ⚠️ Improve a skill
>
> Fill in the angle brackets:
>
> ```text
> In the last factory run, <which station> did this wrong: <what happened>. Read its SKILL.md in .agents/skills and the run in factory/done/. Propose the smallest change to the skill's steps or rules that would have prevented it. Show me the change before you make it.
> ```
>
> **Check:** only approve specific changes. "Be more careful" isn't an instruction; "try every rule in `README.md` as an attack" is. Rerun the station in a new chat and compare.

### Option B: Add a station

> ```
>  ____ _____ ___  ____     ___  ____
> / ___|_   _/ _ \|  _ \   ( _ )| __ )
> \___ \ | || | | | |_) |  / _ \|  _ \
>  ___) || || |_| |  __/  | (_) | |_) |
> |____/ |_| \___/|_|      \___/|____/
> ```
>
> ### ⚠️ Add a new station
>
> ```text
> $skill-creator Make a new skill at .agents/skills/factory-grill/SKILL.md. It runs before factory-intake. It interviews me about a feature idea one question at a time, waits for each answer, and stops when it could write acceptance criteria. Then it writes a 5-line summary I can give to $factory-intake. Follow the format of the other factory skills in .agents/skills.
> ```
>
> **Check:** read the `description`. Then open a new chat and try it:
>
> ```text
> $factory-grill I want a two-player mode where two snakes share the board.
> ```

More ideas: `factory-docs` (update the README after a build), `factory-retro` (read `factory/done/` and find the skill that keeps letting bugs through).

---

```
__      _____    _   ___   _   _ ___
\ \    / / _ \  /_\ | _ \ | | | | _ \
 \ \/\/ /|   / / _ \|  _/ | |_| |  _/
  \_/\_/ |_|_\/_/ \_\_|    \___/|_|
```

## Wrapping up

- A **model** answers one question. A **harness** adds context, a conversation, and the ability to run things. That's an **agent**.
- **`AGENTS.md`** is for every task. **Skills** are for a kind of task.
- One chat guesses, grades its own homework, and forgets. A **factory** fixes all three: stations, people at the gates, and a brain.

Habits for any agent:

- Say how you'll know it's done. "Make it faster" is a wish; "under 50 ms" is a goal.
- Point at an example: "write the test like the ones in `tests/test_game.py`."
- Read the diff. Watch for deleted code, new dependencies, and weaker tests.
- Keep runs small.

**Take it home:** copy `.agents/skills/`, `AGENTS.md`, `BRAIN.md`, and `factory/` into your own project. The workshop key stops working after today; to keep using Codex, run `codex logout`, then `codex login`.

Questions?
