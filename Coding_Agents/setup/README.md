# Workshop Setup (Windows and Mac)

Run the setup script for your computer **before** the workshop. It takes about 10 minutes.

| Computer | Script | Command (run in the `setup` folder) |
| --- | --- | --- |
| Windows 10 or 11 | `setup.ps1` | `powershell -ExecutionPolicy Bypass -File setup.ps1` |
| Mac | `setup.sh` | `bash setup.sh` |

Step-by-step: [Windows](#windows) or [Mac](#mac).

## What the scripts do

Both scripts install the same tools and skip each one you already have. You can run them again at any time. Neither one needs an admin password.

| Tool | Why you need it | Required |
| --- | --- | --- |
| git | Lets the agent see what it changed (the audit skill runs `git diff`) | yes |
| uv | Python package and project manager | yes |
| Python 3.12 | Installed by uv, not from python.org. The script also makes the `python` command (Mac: `python3`) use it. | yes |
| Codex CLI | The coding agent we use in the workshop | yes |
| VS Code | The editor. Installed for your user only. | yes |
| Codex for VS Code | The Codex panel inside VS Code (extension `openai.chatgpt`) | yes |
| Python packages | `pygame-ce` and `pytest`, from the project's `requirements.txt`, installed in the project's `.venv` | yes |

Then the script sets up Codex:

- **Settings:** it makes `.codex/config.toml` in your user folder (or in `CODEX_HOME`, if you set it) with workshop defaults. Codex uses the `gpt-6-luna` model with `medium` reasoning, which is fast and cheap. Codex can edit files in your project and use the internet (for example `uv pip install`), and asks before it does anything outside the project. If you already have this file, the script does not change it, and tells you the lines to add.
- **Workshop project:** it copies the workshop project to `agent-factory` in your user folder, with the [factory skills](../skills/README.md) in its `.agents/skills` folder, where Codex finds them. It makes the folder a git repo with a first commit, and installs the packages in `requirements.txt` into the project's `.venv`. If the folder is already there, the script does not replace it.
- **Sign-in:** if you are not signed in, the script signs you in with the **workshop key**, so you do not need an OpenAI account. The key is in the `workshop-key.txt` file in the zip. If the file is not there, the script asks you to paste the key from the Teams post. To use your own ChatGPT account instead, press Enter, and a browser window opens to sign in. The terminal and VS Code share the same sign-in. If you are already signed in, nothing changes.

Options, for both scripts:

| Windows | Mac | What it does |
| --- | --- | --- |
| `-Check` | `--check` | Shows what is installed and if you are signed in to Codex. It does not change anything. |
| `-Yes` | `--yes` | Does not ask questions. It still signs in with `workshop-key.txt` if the file is there. |

If git does not know your name and email, the script asks for them. git needs them to save commits.

---

## Windows

### Run it

1. Download the workshop zip file. Right-click it and choose **Extract All**.
2. Open the extracted folder. It has `setup`, `skills`, and `starter` folders next to each other, and a `workshop-key.txt` file. Open the `setup` folder.
3. Click the address bar of the File Explorer window, type `powershell`, and press Enter. A PowerShell window opens in that folder. Do not use "Run as Administrator".
4. Paste this and press Enter:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Run it from the extracted folder, not from inside the zip file. If you do, the script cannot find the workshop project or the key. Do not double-click `setup.ps1`: that opens it in Notepad. If you use right-click > "Run with PowerShell", the script waits for Enter at the end so you can read the result.

Windows can show a "Do you want to allow this app to make changes" prompt for git. Click **Yes**. If the script seems stuck, look for a flashing shield icon in the taskbar: the prompt can open behind other windows.

On Windows, Codex keeps to the project folder with a Windows sandbox that does not need admin rights. The script turns it on in `config.toml`.

### After the script

The last lines tell you if you are ready. "Not ready" lists the steps that failed or the required tools that are missing. If sign-in fails, it shows as a warning and do not stop you. "Almost ready" means you still need to sign in: run the script again.

1. Close PowerShell. Open the `setup` folder again and open a new PowerShell window the same way (address bar, type `powershell`). New tools are not found in the old window.
2. Run `powershell -ExecutionPolicy Bypass -File setup.ps1 -Check`. All required rows must show `[OK]`, and `Codex sign-in` should show `[OK]` too. The script also prints this command with the full path at the end, so you can run it from any folder.
3. Open the workshop project in VS Code, then click the Codex icon in the side bar (or open the Command Palette and run **Codex: Open Codex Sidebar**):

```powershell
code $HOME\agent-factory
```

4. Ask Codex: `What does this project do? Run the tests.` If it answers and the tests pass, you are ready.

You can also use Codex in the terminal: `cd $HOME\agent-factory`, then `codex`.

### If a step fails on Windows

The script continues after a failed step and lists the failed steps at the end. Fix the problem, then run the script again.

| Problem | Fix |
| --- | --- |
| `winget was not found` | Install **App Installer** from the Microsoft Store. |
| A school or work laptop blocks installs | Use a personal laptop, or install each tool from its website: [git](https://git-scm.com/download/win), [uv](https://docs.astral.sh/uv/getting-started/installation/), [Codex](https://github.com/openai/codex), [VS Code](https://code.visualstudio.com/). |
| `codex` is not found after the script | Open a new PowerShell window. If it is still missing, run the setup script again. |
| `Packages not installed` | In PowerShell, run `cd $HOME\agent-factory`, then `uv venv`, then `uv pip install -r requirements.txt`. |
| Codex is not signed in | Run the setup script again. It signs in with the workshop key. |
| Codex says the model is not allowed, or `It does not use gpt-6-luna` | You already had a Codex config file. Add the two lines the script printed (`model = ...` and `model_reasoning_effort = ...`) at the **top** of your `config.toml`. Or start Codex with `codex -m gpt-6-luna`. |
| `code` is not found | Open a new PowerShell window. Or open VS Code from the Start menu and use **File > Open Folder** to open `agent-factory` in your user folder. |
| The Codex icon is not in VS Code | Open the Extensions view (Ctrl+Shift+X), search for **Codex**, and install the one from OpenAI. |
| `running scripts is disabled on this system` when you run `setup.ps1` | Use the full command above. It starts with `powershell -ExecutionPolicy Bypass`. If you used that command and it still fails, your organization blocks scripts on this laptop. Use a personal laptop. |
| `Python was not found` or the Microsoft Store opens when you run `python` | Open **Settings > Apps > Advanced app settings > App execution aliases** and turn off `python.exe` and `python3.exe`. Then open a new window. |
| `It has no Windows sandbox setting` | You already had a Codex config file, so the script did not change it. Open your `config.toml` (the path is in the message above it) in Notepad and add the two lines the script printed at the end of the file. |
| `Could not reach astral.sh` | Your network blocks it, or you are offline. Try another network (for example a phone hotspot), then run the script again. |
| `No starter or skills folder next to the setup folder` | Extract the zip file first, then run `setup.ps1` from the extracted `setup` folder. The `setup`, `skills`, and `starter` folders must be next to each other. |
| `Could not copy skill` | Close any program that has the skill files open, then run the script again. It copies the skills that did not finish. |

Bring the output of `setup.ps1 -Check` to the workshop if something still does not work.

---

## Mac

### Run it

1. Download the workshop zip file and double-click it. macOS unzips it into a `coding-agents-workshop` folder next to the zip (usually in **Downloads**).
2. Open the **Terminal** app (press Cmd+Space, type `Terminal`, press Enter).
3. Type `bash ` (with a space after it), then drag `setup.sh` from the unzipped `setup` folder into the Terminal window. Press Enter. If the folder is in Downloads, you can paste this instead:

```bash
bash ~/Downloads/coding-agents-workshop/setup/setup.sh
```

Do not use `sudo`. Everything goes in your user folder: VS Code in `~/Applications`, the command-line tools in `~/.local/bin`.

If your Mac does not have git yet, a window asks to install Apple's **Command Line Tools**. Click **Install** and wait a few minutes. The script waits for it, then continues.

The script adds `~/.local/bin` to your PATH in `~/.zprofile` (or `~/.bash_profile` if you use bash), so new Terminal windows find the tools.

### After the script

The last lines tell you if you are ready, the same way as on Windows.

1. Quit Terminal (Cmd+Q) and open it again. New tools are not found in the old window.
2. Run the check. All required rows must show `[OK]`, and `Codex sign-in` should show `[OK]` too. The script prints this command with the full path at the end.

```bash
bash ~/Downloads/coding-agents-workshop/setup/setup.sh --check
```

3. Open the workshop project in VS Code, then click the Codex icon in the side bar (or open the Command Palette with Cmd+Shift+P and run **Codex: Open Codex Sidebar**):

```bash
code ~/agent-factory
```

4. Ask Codex: `What does this project do? Run the tests.` If it answers and the tests pass, you are ready.

You can also use Codex in the terminal: `cd ~/agent-factory`, then `codex`.

### If a step fails on Mac

| Problem | Fix |
| --- | --- |
| `No such file or directory` when you run `bash ...setup.sh` | The folder is not where the command says. Type `bash ` and drag `setup.sh` into Terminal instead. |
| `$'\r': command not found` | The script has Windows line endings. Download the zip again from Teams and tell a facilitator. |
| The Command Line Tools window did not open, or the install failed | Run `xcode-select --install`, finish the install, then run the script again. |
| `codex`, `uv`, or `code` is not found after the script | Quit Terminal and open it again. If it is still missing, check that `~/.zprofile` has the line the script added (`export PATH="$HOME/.local/bin:...`). |
| macOS says VS Code "cannot be opened" or "is damaged" | Open **System Settings > Privacy & Security**, scroll down, and click **Open Anyway** for Visual Studio Code. |
| Codex is not signed in | Run the setup script again. It signs in with the workshop key. |
| Codex says the model is not allowed, or `It does not use gpt-6-luna` | You already had a Codex config file. Add the two lines the script printed at the **top** of `~/.codex/config.toml`. Or start Codex with `codex -m gpt-6-luna`. |
| The Codex icon is not in VS Code | Open the Extensions view (Cmd+Shift+X), search for **Codex**, and install the one from OpenAI. |
| `Packages not installed` | In Terminal, run `cd ~/agent-factory && uv venv && uv pip install -r requirements.txt`. |
| `Could not reach astral.sh` | Your network blocks it, or you are offline. Try another network, then run the script again. |
| `No starter or skills folder next to the setup folder` | Unzip the zip file first, then run `setup.sh` from the unzipped `setup` folder. The `setup`, `skills`, and `starter` folders must be next to each other. |

Bring the output of `setup.sh --check` to the workshop if something still does not work.
