# Workshop Setup (Windows)

`setup.ps1` installs and sets up the tools for the Coding Agents workshop. Run it **before** the workshop. It takes about 10 minutes.

| Tool | Why you need it | Required |
| --- | --- | --- |
| git | Lets the agent see what it changed (the review skill runs `git diff`) | yes |
| uv | Python package and project manager | yes |
| Python 3.12 | Installed by uv, not from python.org. The script also makes the `python` command work. | yes |
| Node.js LTS | Web projects and many MCP servers (`npx`) | yes |
| Codex CLI | The coding agent we use in the workshop | yes |
| ruff | Python linter that agents can run to check their work | no |

The script also lets PowerShell run local scripts for your user only (`RemoteSigned`). It asks first, unless you use `-Yes`. If your organization controls this policy, the script cannot change it; type `npm.cmd` and `npx.cmd` instead of `npm` and `npx` (and `codex.cmd` if Codex came from npm). Without this, `npm` and `npx` do not work in a new PowerShell window. It does not need admin rights. To do it yourself, run this in PowerShell (not as Administrator):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Answer **Y**. Then open a new PowerShell window.

Then the script sets up Codex:

- **Settings:** it makes `%USERPROFILE%\.codex\config.toml` (or `config.toml` in `%CODEX_HOME%`, if you set it) with workshop defaults. With these settings, Codex can edit files in your project and use the internet (for example `npm install` or `uv sync`), and asks before it does anything outside the project. Codex keeps to this with a Windows sandbox that does not need admin rights. If you already have this file, the script does not change it.
- **Skills:** it copies the [workshop skills](../skills/README.md) to `%USERPROFILE%\.agents\skills`, where Codex finds them. It does not replace a skill that is already there.
- **Sign-in:** if you are not signed in, it asks to sign you in. A new window opens, then your browser. Sign in with your ChatGPT account. If sign-in fails, that window shows the error and waits for Enter. If it gets stuck, close the new window and run `codex login` later.

The script skips each tool and setting that you already have. You can run it again at any time.

## Run it

1. Download the workshop zip file. Right-click it and choose **Extract All**.
2. Open the extracted folder. It has a `setup` folder and a `skills` folder next to each other. Open the `setup` folder.
3. Click the address bar of the File Explorer window, type `powershell`, and press Enter. A PowerShell window opens in that folder. Do not use "Run as Administrator".
4. Paste this and press Enter:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Run it from the extracted folder, not from inside the zip file. If you do, the script cannot find the workshop skills. Do not double-click `setup.ps1`: that opens it in Notepad. If you use right-click > "Run with PowerShell", the script waits for Enter at the end so you can read the result.

Options:

| Option | What it does |
| --- | --- |
| `-Check` | Shows what is installed and if you are signed in to Codex. It does not change anything. |
| `-Yes` | Does not ask questions. It skips Codex sign-in. |

Windows can show a "Do you want to allow this app to make changes" prompt for git or Node.js. Click **Yes**. If the script seems stuck, look for a flashing shield icon in the taskbar: the prompt can open behind other windows.

If git does not know your name and email, the script asks for them (with `-Yes`, it only shows the commands to set them). git needs them to save commits.

## After the script

The last lines tell you if you are ready. "Not ready" lists the steps that failed or the required tools that are missing. Optional steps that fail (ruff, sign-in) show as a warning and do not stop you. "Almost ready" means you still need to run `codex login`.

1. Close PowerShell. Open the `setup` folder again and open a new PowerShell window the same way (address bar, type `powershell`). New tools are not found in the old window.
2. Run `powershell -ExecutionPolicy Bypass -File setup.ps1 -Check`. All required rows must show `[OK]`, and `Codex sign-in` should show `[OK]` too. The script also prints this command with the full path at the end, so you can run it from any folder.
3. Make a folder for your work and start Codex in it:

```powershell
mkdir $HOME\workshop
cd $HOME\workshop
codex
```

## If a step fails

The script continues after a failed step and lists the failed steps at the end. Fix the problem, then run the script again.

| Problem | Fix |
| --- | --- |
| `winget was not found` | Install **App Installer** from the Microsoft Store. |
| A school or work laptop blocks installs | Use a personal laptop, or install each tool from its website: [git](https://git-scm.com/download/win), [uv](https://docs.astral.sh/uv/getting-started/installation/), [Node.js](https://nodejs.org), [Codex](https://github.com/openai/codex). |
| `codex` is not found after the script | Open a new PowerShell window. If it is still missing, run `npm.cmd install -g @openai/codex`. |
| Codex is not signed in | Run `codex login`. |
| `running scripts is disabled on this system` when you run `setup.ps1` | Use the full command above. It starts with `powershell -ExecutionPolicy Bypass`. If you used that command and it still fails, your organization blocks scripts on this laptop. Use a personal laptop. |
| `running scripts is disabled on this system` when you run `npm`, `npx`, or `codex` | Run the setup script again and answer **Y** to the script policy question, or run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` yourself. If your organization controls the policy, type `npm.cmd` and `npx.cmd` instead. |
| `Python was not found` or the Microsoft Store opens when you run `python` | Open **Settings > Apps > Advanced app settings > App execution aliases** and turn off `python.exe` and `python3.exe`. Then open a new window. |
| `It has no Windows sandbox setting` | You already had a Codex config file, so the script did not change it. Open your `config.toml` (the path is in the message above it) in Notepad and add the two lines the script printed at the end of the file. |
| `Could not reach astral.sh` | Your network blocks it, or you are offline. Try another network (for example a phone hotspot), then run the script again. |
| `Workshop skills not copied` | Extract the zip file first, then run `setup.ps1` from the extracted `setup` folder. The `setup` and `skills` folders must be next to each other. |
| `Could not copy skill` | Close any program that has the skill files open, then run the script again. It copies the skills that did not finish. |

Bring the output of `setup.ps1 -Check` to the workshop if something still does not work.
