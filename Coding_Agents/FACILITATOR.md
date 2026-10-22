# Facilitator Guide

How to run the Coding Agents workshop for about 50 people with one shared OpenAI API key, so nobody needs their own account.

## 1. Make the workshop key (a week before)

Use a separate project so the key can only spend what you put in, and only on the cheap model.

1. On [platform.openai.com](https://platform.openai.com), make a new project, for example `MAIC Workshop`.
2. **Billing:** buy $100 of prepaid credits. Turn **off** auto-reload (it is on by default). When the credits run out, requests stop.
3. **Project spend limit:** Project settings > Limits > Spend > set $100 and turn on **Enforce a hard limit**. Requests then fail with `project_spend_limit_exceeded` instead of spending more.
4. **Model allowlist:** Project settings > Limits > Model usage > allow only `gpt-6-luna`. A student who changes the model gets an error instead of spending 20x more.
5. **Rate limits:** on the same Limits page, check the tokens-per-minute (TPM) limit for `gpt-6-luna`. 50 agents at once can send several million tokens per minute. A new account has low limits that go up with your usage tier, which depends on how much you have paid and how long ago. Buy the credits early. If the limit is low, have people work in pairs. Codex retries when it hits a rate limit, so a low limit makes it slow, not broken.
6. Make an API key in the project. Save it in a text file, one line, outside this repo.
7. Test it on your own laptop: run the setup with the key, open the project, and ask Codex to run the tests. This also proves your organization can use `gpt-6-luna` with an API key.

**Cost estimate:** `gpt-6-luna` costs $0.10 per million input tokens ($0.01 cached) and $0.50 per million output tokens. Codex sends the conversation again on each step, and most of it is cached. A guess for one person's hour of hands-on work: about 5 million input tokens and 100,000 output tokens, which is about $0.15. With 50 people that is under $10. Even at 10 times that, $100 is enough. Check the Usage page after the test run in step 7 and scale it up.

## 2. Build and share the zip (5 days before)

```powershell
cd Coding_Agents\setup
powershell -ExecutionPolicy Bypass -File make-zip.ps1 -KeyFile C:\path\to\workshop-key.txt
```

This makes `Coding_Agents\dist\coding-agents-workshop.zip` with `setup` (both `setup.ps1` for Windows and `setup.sh` for Mac), `skills`, `starter`, the workshop guide (`coding-agents-workshop.md`), and `workshop-key.txt`. The `dist` folder and key files are in `.gitignore`. The script writes `setup.sh` with Mac line endings even if your git checkout changed them.

Anyone who has the zip can read the key. The hard limit and the model allowlist are what keep it safe. To be more careful, build the zip without `-KeyFile` and post the key in the Teams chat on workshop day: both setup scripts ask for it.

Teams post:

> **Coding Agents workshop: set up before <day>** (about 10 minutes, Windows or Mac)
>
> **Windows**
> 1. Download the zip below. Right-click it > **Extract All**.
> 2. Open the extracted `setup` folder. Click the address bar, type `powershell`, press Enter.
> 3. Paste `powershell -ExecutionPolicy Bypass -File setup.ps1` and press Enter. Answer the questions.
> 4. When it says **Ready**, open a new PowerShell window and run `code $HOME\agent-factory`.
>
> **Mac**
> 1. Download the zip below and double-click it.
> 2. Open **Terminal** (Cmd+Space, type Terminal).
> 3. Type `bash ` (with a space), drag `setup.sh` from the unzipped `setup` folder into the window, and press Enter. Answer the questions.
> 4. When it says **Ready**, quit Terminal, open it again, and run `code ~/agent-factory`.
>
> You do not need an OpenAI account. If something fails, reply here with a screenshot of the last lines.

Test both scripts before you post: `setup.ps1` on a Windows laptop and `setup.sh` on a Mac.

## 3. Run of show (60 minutes)

| Time | Workshop section | What happens |
| --- | --- | --- |
| 0:00 | Prereqs | Doors open. Everyone runs the setup check (`-Check` or `--check`). Helpers fix setups. |
| 0:08 | "Models, harnesses, and agents" | You present, they read. The key word is **harness**: plain chat answers one question; the harness adds context, the conversation, and tools; an agent loops on its own. |
| 0:14 | "AGENTS.md", STOP 1 | Everyone opens the project and the Codex panel. The first prompt shows Codex already knows the rules in `AGENTS.md`. |
| 0:18 | "Skills", STOP 2 | Codex lists the four skills; people type `$` to see them. |
| 0:21 | "Why one chat isn't enough", "The software factory" | You present. No prompts. |
| 0:26 | Station 1, STOP 3 | Intake |
| 0:29 | Station 2, STOP 4 | Shape. People approve the spec. |
| 0:35 | Station 3, STOP 5 | Build. Watch the red-green loop. |
| 0:41 | Station 4, STOP 6 and 7 | People write their own audit skill from the template, then run it in a new chat and judge the findings. |
| 0:48 | Station 5, STOP 8 | Close. People approve the lessons. Ask whose audit caught the classic bug. |
| 0:51 | "Make the factory yours", STOP 9A or 9B | Improve a skill or add a station |
| 0:57 | "Wrapping up" | Take-home, next steps (MCP, parallel agents, where to learn more), and Q&A |

The student guide has no times in it on purpose. Run it live on the projector, one STOP box ahead of the room. Slides with an amber BACK TO CODEX badge are the STOP boxes: pause and let everyone send the prompt.

## 4. The starter project and the classic bug

The starter is only the bones of Snake: `snake/game.py` and `snake/app.py` have TODOs, there's one test, and the packages are in `requirements.txt` (`pygame-ce`, `pytest`). The first ticket (STOP 3) has the factory build the whole game from the rules in `README.md`. In this repo, `.gitignore` keeps anything an agent writes on top of the bones out of git, so you can rehearse in `starter/` without committing it.

Watch for the classic Snake bug, which agents (and people) write all the time. `README.md` says the snake can never turn back on itself, "no matter how fast you press the keys." The usual code compares a new turn with the *pending* direction instead of the direction the snake last moved. Moving right, press **up then left** before the next tick: both turns are allowed, and the snake reverses into its own neck. A test that presses one key passes anyway.

To check someone's game, run `uv run python -m snake --speed 3` and tap up and left quickly. The fix is to compare against the direction of the last actual move (or queue the turns). The STOP 6 prompt tells audit skills to check the README rules, and the shape skill asks for "doing two things at once" edge cases. After STOP 8, ask the room: whose audit caught it? If a factory missed it, that's the lesson to save in `BRAIN.md` and the instruction to improve in STOP 9A.

## 5. The audit skill people write

Students don't get an audit skill. In STOP 6 they write one with Codex from `factory/skill-template.md`. The prompt fixes the interface the other stations depend on: it reads `factory/current/SPEC.md` and the diff, never fixes code, refuses to run in the builder's chat, and writes `factory/current/REVIEW.md` ending in `Verdict: PASS` or `Verdict: REVISE`. The rest is theirs, including what Codex asks them a good reviewer should check.

A reference answer is in [`solutions/factory-audit/SKILL.md`](solutions/factory-audit/SKILL.md). It isn't in the zip. If someone's skill is badly broken and they're falling behind, copy that folder into their `.agents/skills/` so they can keep going.

## 6. After the workshop

1. Revoke the API key that night. The key stays in each laptop's Codex login, so revoking is the only way to stop it.
2. Tell people how to keep using Codex with their own account: `codex logout`, then `codex login`.
3. Check the Usage page and write down the real cost per person for next time.
