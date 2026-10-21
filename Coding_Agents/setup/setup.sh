#!/usr/bin/env bash
#
# Coding Agents workshop setup for macOS. The Windows version is setup.ps1.
#
# Installs: git (Apple's Command Line Tools), uv, Python 3.12 (through uv),
# the Codex CLI, VS Code, and the Codex VS Code extension. The project's Python
# packages come from its requirements.txt.
# Then it sets up Codex (workshop defaults and sign-in) and copies the workshop
# project, with the factory skills, to ~/agent-factory.
# Everything goes in your user folder, so it needs no admin password.
# Safe to run more than one time: it skips tools and settings that are already
# there.
#
# Sign-in: if a workshop-key.txt file is next to the setup folder, Codex signs in
# with that API key, so students do not need an account. Without the file, the
# script asks for a key, or signs in with a ChatGPT account.
#
# Usage (in Terminal):
#   bash setup.sh            install everything
#   bash setup.sh --check    only show what is installed
#   bash setup.sh --yes      do not ask
#
# Written for the bash 3.2 that comes with macOS: no associative arrays or
# other bash 4 features.

PYTHON_VERSION=3.12
MODEL=gpt-6-luna
REASONING_EFFORT=medium
CODEX_EXTENSION=openai.chatgpt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SKILLS_SRC="$ZIP_ROOT/skills"
STARTER_SRC="$ZIP_ROOT/starter"
KEY_FILE="$ZIP_ROOT/workshop-key.txt"
PROJECT_DIR="$HOME/agent-factory"
# Repo skills: Codex finds them in .agents/skills when it runs in the project.
SKILLS_DEST="$PROJECT_DIR/.agents/skills"
CODEX_HOME_DIR="${CODEX_HOME:-$HOME/.codex}"
CODEX_CONFIG="$CODEX_HOME_DIR/config.toml"
LOCAL_BIN="$HOME/.local/bin"
VSCODE_USER_APP="$HOME/Applications/Visual Studio Code.app"
PROFILE_MARK='# Added by the AI Club workshop setup'
PROFILE_LINE='export PATH="$HOME/.local/bin:$PATH"'

CHECK=0
YES=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK=1 ;;
        --yes | -y) YES=1 ;;
        -h | --help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (use --check or --yes)"; exit 2 ;;
    esac
done

# uv, Codex, and the code command all go in this folder.
export PATH="$LOCAL_BIN:$PATH"

# Theme: MAIC purple, blue, and black. Warnings stay yellow and errors red.
say()  { printf '\033[35m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[36m  [OK] %s\033[0m\n' "$*"; }
warn() { printf '\033[33m  [!]  %s\033[0m\n' "$*"; }
fail() { printf '\033[31m  [X]  %s\033[0m\n' "$*"; }
have() { hash -r; command -v "$1" >/dev/null 2>&1; }

# Reads from the terminal, so questions work even when stdin is a pipe.
ask() { printf '  %s' "$1" >/dev/tty; IFS= read -r REPLY </dev/tty; }

confirm() {
    [ "$YES" = 1 ] && return 0
    ask "$1 [Y/n] "
    case "$REPLY" in [nN]*) return 1 ;; esac
    return 0
}

first_line() { sed -n '1p' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'; }

# /usr/bin/git and /usr/bin/python3 are stubs that open an install window until
# Apple's Command Line Tools are installed. Do not run them before that.
clt_ok() { xcode-select -p >/dev/null 2>&1; }
git_ok() { have git && { clt_ok || [ "$(command -v git)" != /usr/bin/git ]; }; }

python_version() {
    # The python3 a new Terminal window runs. It must be uv's 3.12, which comes
    # before /usr/bin in PATH.
    have python3 || return 0
    if [ "$(command -v python3)" = /usr/bin/python3 ] && ! clt_ok; then return 0; fi
    python3 --version 2>/dev/null | first_line
}

code_cmd() {
    # VS Code can be installed without the code command on PATH, so also look
    # in the app folders.
    if have code; then command -v code; return 0; fi
    local app
    for app in "/Applications/Visual Studio Code.app" "$VSCODE_USER_APP"; do
        if [ -x "$app/Contents/Resources/app/bin/code" ]; then
            echo "$app/Contents/Resources/app/bin/code"
            return 0
        fi
    done
    return 1
}

vscode_version() { local code; code="$(code_cmd)" && "$code" --version 2>/dev/null | first_line; }

codex_extension_ok() {
    local code
    code="$(code_cmd)" || return 1
    "$code" --list-extensions 2>/dev/null | grep -qx "$CODEX_EXTENSION"
}

read_workshop_key() {
    # The key file is in the zip that facilitators share. It is never in git.
    [ -f "$KEY_FILE" ] || return 0
    local key
    key="$(tr -d '[:space:]' <"$KEY_FILE")"
    case "$key" in
        sk-*) echo "$key" ;;
        *) warn "$KEY_FILE does not look like an OpenAI API key (it should start with sk-)." >&2 ;;
    esac
}

profile_file() {
    # Terminal and VS Code start login shells, which read these files.
    case "$(basename "${SHELL:-/bin/zsh}")" in
        bash) echo "$HOME/.bash_profile" ;;
        *) echo "$HOME/.zprofile" ;;
    esac
}

FAILED_STEPS=""
FAILED_OPTIONAL=""
run_step() {
    # run_step "Label" function_name [optional]
    say "$1"
    if ! "$2"; then
        if [ "${3:-}" = optional ]; then
            warn "$1 did not finish (optional)."
            FAILED_OPTIONAL="${FAILED_OPTIONAL:+$FAILED_OPTIONAL, }$1"
        else
            fail "$1 did not finish. See the messages above."
            FAILED_STEPS="${FAILED_STEPS:+$FAILED_STEPS, }$1"
        fi
    fi
}

show_logo() {
    # The MAIC logo: a network-shaped brain. Columns go blue -> cyan -> purple,
    # like the club logo.
    local colors=(34 34 36 36 37 35 35 35 35) line i idx
    printf '\n'
    while IFS= read -r line; do
        printf '  '
        i=0
        while [ "$i" -lt "${#line}" ]; do
            idx=$((i / 5))
            [ "$idx" -gt 8 ] && idx=8
            printf '\033[%sm%s' "${colors[$idx]}" "${line:$i:1}"
            i=$((i + 1))
        done
        printf '\033[0m\n'
    done <<'EOF'
            o-------------------O
          ./ \               _.-'  \.
       (@)    o------o----v-'       O
       / \   / \    /      \       /|
      O---o-'   \  /        o-----o |
       \   \     o'   M A I C   \   |
        \   |>   |               o--O
         O--+----+---o--------o-'   /
          '.|        |       /  \  /
            o--------o------o----O'
                               \ |
                                \|
                                 o
EOF
    printf '\n   \033[36mMSOE AI Club\033[90m  |  \033[35mCoding Agents Workshop setup\033[0m\n\n'
}

# ---------- version report ----------

MISSING_REQUIRED=0
SIGNED_IN=0
report_row() {
    # report_row "label" "version or empty" required|optional
    if [ -n "$2" ]; then
        printf '\033[36m  [OK] %-16s %s\033[0m\n' "$1" "$2"
    elif [ "$3" = required ]; then
        printf '\033[31m  [X]  %-16s missing\033[0m\n' "$1"
        MISSING_REQUIRED=1
    else
        printf '\033[33m  [-]  %-16s missing (optional)\033[0m\n' "$1"
    fi
}

show_report() {
    MISSING_REQUIRED=0
    SIGNED_IN=0
    local py="" status="" signin=""
    if have uv && uv python find "$PYTHON_VERSION" >/dev/null 2>&1; then py=installed; fi
    say 'Workshop tools'
    report_row 'git'             "$(git_ok && git --version | sed 's/git version //')" required
    report_row 'uv'              "$(have uv && uv --version | sed 's/^uv //')" required
    report_row "Python $PYTHON_VERSION" "$py" required
    report_row 'python3 command' "$(python_version | grep "^Python $PYTHON_VERSION")" required
    report_row 'codex'           "$(have codex && codex --version 2>/dev/null | first_line)" required
    report_row 'VS Code'         "$(vscode_version)" required
    report_row 'Codex extension' "$(codex_extension_ok && echo "$CODEX_EXTENSION")" required
    report_row 'project'         "$([ -f "$PROJECT_DIR/AGENTS.md" ] && echo "$PROJECT_DIR")" required
    report_row 'packages'        "$([ -d "$PROJECT_DIR/.venv" ] && echo 'installed in .venv')" required
    if have codex; then
        # "Logged in using an API key - sk-...1234" or "Logged in using ChatGPT".
        # The key is masked in this output.
        if status="$(codex login status 2>&1)"; then
            SIGNED_IN=1
            signin="$(echo "$status" | first_line | sed 's/^Logged in using //')"
        fi
    fi
    report_row 'Codex sign-in'   "$signin" optional

    if git_ok; then
        local name email
        name="$(git config --global user.name)"
        email="$(git config --global user.email)"
        if [ -n "$name" ] && [ -n "$email" ]; then ok "git identity: $name <$email>"
        else warn 'git identity is not set (git config --global user.name / user.email)'; fi
    fi
}

RUN_COMMAND="bash \"$SCRIPT_DIR/setup.sh\""
CHECK_COMMAND="$RUN_COMMAND --check"

show_logo

if [ "$CHECK" = 1 ]; then
    show_report
    echo
    if [ "$MISSING_REQUIRED" = 1 ]; then
        fail 'Some required tools are missing. To install them, run:'
        echo "  $RUN_COMMAND"
        exit 1
    fi
    if [ "$SIGNED_IN" = 1 ]; then ok "Ready for the workshop. Open the project with: code ~/agent-factory"
    else warn 'Codex is not signed in. Run the setup script again, or run: codex login'; fi
    exit 0
fi

# ---------- install steps ----------

say "Coding Agents workshop setup (macOS, $(uname -m))"
echo "  This installs: git, uv, Python $PYTHON_VERSION, Codex CLI, VS Code, Codex for VS Code."
echo '  Everything goes in your user folder. Tools you already have are skipped.'
if ! curl -fsS --max-time 10 -o /dev/null https://astral.sh; then
    warn 'Could not reach astral.sh (offline, or the network blocks it). The uv and Python steps can fail. Other steps continue.'
fi
confirm 'Continue?' || exit 0

step_git() {
    # Agents use git to see what they changed (the audit skill runs git diff).
    if ! git_ok; then
        echo "  git comes with Apple's Command Line Tools. A window opens: click Install, and wait."
        xcode-select --install >/dev/null 2>&1
        local waited=0
        until clt_ok; do
            sleep 10
            waited=$((waited + 10))
            if [ "$waited" -ge 1800 ]; then
                fail 'The Command Line Tools did not finish in 30 minutes. Run xcode-select --install, then run this again.'
                return 1
            fi
        done
    fi
    git_ok || return 1
    ok "$(git --version)"

    # git needs a name and email before it can save a commit.
    local name email i
    name="$(git config --global user.name)"
    email="$(git config --global user.email)"
    if [ -n "$name" ] && [ -n "$email" ]; then ok "git identity: $name <$email>"; return 0; fi
    if [ "$YES" = 1 ]; then
        warn 'git identity is not set. Run: git config --global user.name "Your Name"'
        warn 'and: git config --global user.email "you@example.com"'
        return 0
    fi
    echo '  git needs a name and email to save commits.'
    i=0
    while [ -z "$name" ] && [ "$i" -lt 3 ]; do ask 'Your name: '; name="$REPLY"; i=$((i + 1)); done
    [ -n "$name" ] && git config --global user.name "$name"
    i=0
    while [ -z "$email" ] && [ "$i" -lt 3 ]; do
        ask 'Your email: '
        case "$REPLY" in
            *[!\ ]@[!\ ]*) email="$REPLY" ;;
            *) warn 'That is not an email address (example: name@example.com).' ;;
        esac
        i=$((i + 1))
    done
    [ -n "$email" ] && git config --global user.email "$email"
    if [ -z "$name" ] || [ -z "$email" ]; then
        warn 'git identity is not set. Run: git config --global user.name "Your Name"'
        warn 'and: git config --global user.email "you@example.com"'
    fi
    return 0
}

step_path() {
    # New Terminal windows must find the tools in ~/.local/bin.
    local profile
    profile="$(profile_file)"
    if [ -f "$profile" ] && grep -qF "$PROFILE_MARK" "$profile"; then
        ok "$profile already adds ~/.local/bin to PATH"
        return 0
    fi
    printf '\n%s\n%s\n' "$PROFILE_MARK" "$PROFILE_LINE" >>"$profile" || return 1
    ok "Added ~/.local/bin to PATH in $profile"
}

step_uv() {
    if have uv; then ok "uv $(uv --version | sed 's/^uv //')"; return 0; fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    have uv
}

step_python() {
    have uv || return 1
    local version
    version="$(python_version)"
    case "$version" in "Python $PYTHON_VERSION".*) ok "python3 command: $version"; return 0 ;; esac
    # --default also adds python and python3 commands to ~/.local/bin.
    if ! uv python install "$PYTHON_VERSION" --default; then
        # Older uv versions need --preview for --default.
        uv python install "$PYTHON_VERSION" --default --preview || return 1
    fi
    version="$(python_version)"
    case "$version" in
        "Python $PYTHON_VERSION".*) ok "python3 command: $version" ;;
        *) fail "python3 is $version, not $PYTHON_VERSION. Open a new Terminal window and run this again."; return 1 ;;
    esac
}

step_codex() {
    if have codex; then ok "$(codex --version 2>/dev/null | first_line)"; return 0; fi
    curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_NON_INTERACTIVE=true sh
    have codex
}

step_vscode() {
    local code tmp
    if ! code="$(code_cmd)"; then
        say '  Installing VS Code to ~/Applications'
        tmp="$(mktemp -d)"
        curl -fL --progress-bar -o "$tmp/vscode.dmg" \
            https://update.code.visualstudio.com/latest/darwin-universal-dmg/stable || { rm -rf "$tmp"; return 1; }
        hdiutil attach -nobrowse -quiet -mountpoint "$tmp/mnt" "$tmp/vscode.dmg" || { rm -rf "$tmp"; return 1; }
        mkdir -p "$HOME/Applications"
        ditto "$tmp/mnt/Visual Studio Code.app" "$VSCODE_USER_APP"
        hdiutil detach -quiet "$tmp/mnt"
        rm -rf "$tmp"
        code="$(code_cmd)" || return 1
    fi
    # The code command, for new Terminal windows.
    if ! have code; then
        mkdir -p "$LOCAL_BIN"
        ln -sf "$code" "$LOCAL_BIN/code"
    fi
    ok "VS Code $(vscode_version)"
}

step_extension() {
    local code
    code="$(code_cmd)" || return 1
    if codex_extension_ok; then ok "Extension $CODEX_EXTENSION already installed"; return 0; fi
    "$code" --install-extension "$CODEX_EXTENSION"
    codex_extension_ok
}

step_codex_settings() {
    # Workshop defaults: the cheap model, and Codex can edit files in the project
    # and use the network (uv pip install), and asks before it does anything outside
    # it. An existing config is never changed.
    if [ -f "$CODEX_CONFIG" ]; then
        ok "$CODEX_CONFIG already exists, not changed"
        local model
        model="$(sed -n 's/^[[:space:]]*model[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$CODEX_CONFIG" | sed -n '1p')"
        if [ "$model" != "$MODEL" ]; then
            warn "It does not use $MODEL. The workshop key only works with $MODEL. Add these lines at the top of it:"
            echo "      model = \"$MODEL\""
            echo "      model_reasoning_effort = \"$REASONING_EFFORT\""
        fi
        return 0
    fi
    mkdir -p "$CODEX_HOME_DIR" || return 1
    cat >"$CODEX_CONFIG" <<EOF
# Made by the AI Club workshop setup script. Edit it as you like.
model = "$MODEL"
model_reasoning_effort = "$REASONING_EFFORT"
approval_policy = "on-request"
sandbox_mode = "workspace-write"

[sandbox_workspace_write]
network_access = true
EOF
    ok "Made $CODEX_CONFIG"
}

step_project() {
    if [ ! -d "$STARTER_SRC" ] || [ ! -d "$SKILLS_SRC" ]; then
        fail "No starter or skills folder next to the setup folder ($ZIP_ROOT)."
        fail 'Unzip the workshop zip file, then run setup.sh from the unzipped setup folder.'
        return 1
    fi
    if [ -f "$PROJECT_DIR/AGENTS.md" ]; then
        ok "$PROJECT_DIR already exists, not changed"
    else
        mkdir -p "$PROJECT_DIR"
        # The /. copies dot files such as .gitignore too.
        cp -R "$STARTER_SRC/." "$PROJECT_DIR/" || return 1
        ok "Copied the workshop project to $PROJECT_DIR"
    fi

    mkdir -p "$SKILLS_DEST"
    local skill name target copy_failed=0
    for skill in "$SKILLS_SRC"/*/; do
        [ -f "$skill/SKILL.md" ] || continue
        name="$(basename "$skill")"
        target="$SKILLS_DEST/$name"
        # SKILL.md is copied last, so a skill that has it is complete. A folder
        # without it is from a copy that did not finish, so it is copied again.
        if [ -f "$target/SKILL.md" ]; then ok "Skill $name already exists, not changed"; continue; fi
        mkdir -p "$target"
        if (cd "$skill" && find . -mindepth 1 -maxdepth 1 ! -name SKILL.md -exec cp -R {} "$target/" \;) &&
            cp "$skill/SKILL.md" "$target/SKILL.md"; then
            ok "Copied skill $name"
        else
            fail "Could not copy skill $name"
            copy_failed=1
        fi
    done

    # The skills read git diffs, so the project starts as a repo with one commit.
    if git_ok && [ ! -d "$PROJECT_DIR/.git" ]; then
        git -C "$PROJECT_DIR" init --quiet
        git -C "$PROJECT_DIR" add -A
        local identity=()
        [ -n "$(git config --global user.name)" ] || identity+=(-c user.name=Workshop)
        [ -n "$(git config --global user.email)" ] || identity+=(-c user.email=workshop@example.com)
        if git -C "$PROJECT_DIR" ${identity[@]+"${identity[@]}"} commit --quiet -m 'Starting point'; then
            ok 'Made a git repo with a first commit'
        else
            warn "Could not make the first commit. In $PROJECT_DIR run: git add -A && git commit -m start"
        fi
    fi

    # The packages in requirements.txt go in the project's .venv, where uv run finds them.
    local packages_ok=0
    if have uv && (cd "$PROJECT_DIR" &&
        { [ -d .venv ] || uv venv --quiet --python "$PYTHON_VERSION"; } &&
        uv pip install --quiet -r requirements.txt); then
        packages_ok=1
        ok 'Packages from requirements.txt installed in .venv'
    else
        fail "Packages not installed. In $PROJECT_DIR run: uv venv && uv pip install -r requirements.txt"
    fi
    [ "$copy_failed" = 0 ] && [ "$packages_ok" = 1 ]
}

step_signin() {
    have codex || return 1
    local status key=""
    if status="$(codex login status 2>&1)"; then
        ok "Signed in to Codex: $(echo "$status" | first_line)"
        return 0
    fi

    key="$(read_workshop_key)"
    if [ -n "$key" ]; then
        echo '  Using the workshop API key from workshop-key.txt.'
    elif [ "$YES" != 1 ]; then
        echo '  If the workshop gave you an API key, paste it here (Cmd+V). It is not shown.'
        echo '  To use your own ChatGPT account instead, press Enter.'
        printf '  Workshop key: ' >/dev/tty
        IFS= read -rs key </dev/tty
        echo
        key="$(printf '%s' "$key" | tr -d '[:space:]')"
        case "$key" in
            '' | sk-*) ;;
            *) warn 'That does not look like an API key (it should start with sk-).'; key="" ;;
        esac
    fi
    if [ -n "$key" ]; then
        # --with-api-key reads stdin, so the key is not in the command line or history.
        # The CLI and the VS Code extension share this sign-in.
        printf '%s\n' "$key" | codex login --with-api-key
        if codex login status >/dev/null 2>&1; then ok 'Signed in to Codex with the workshop key'; return 0; fi
        warn 'Sign-in with the workshop key did not finish. Ask a facilitator.'
        return 1
    fi

    if [ "$YES" = 1 ] || ! confirm 'Sign in to Codex now? Use your ChatGPT account.'; then
        warn 'Not signed in. Later, run: codex login'
        return 0
    fi
    echo '  Your browser opens to sign in. If it gets stuck, press Ctrl+C and run codex login later.'
    codex login </dev/tty
    if codex login status >/dev/null 2>&1; then ok 'Signed in to Codex'; return 0; fi
    warn 'Not signed in. Later, run: codex login'
    return 1
}

run_step 'git' step_git
run_step 'Shell PATH' step_path
run_step 'uv' step_uv
run_step "Python $PYTHON_VERSION" step_python
run_step 'Codex CLI' step_codex
run_step 'VS Code' step_vscode
run_step 'Codex for VS Code' step_extension
run_step 'Codex settings' step_codex_settings
run_step 'Workshop project' step_project
run_step 'Codex sign-in' step_signin optional

echo
show_report
echo

if [ -n "$FAILED_STEPS" ] || [ "$MISSING_REQUIRED" = 1 ]; then
    if [ -n "$FAILED_STEPS" ]; then fail "Not ready. These steps did not finish: $FAILED_STEPS"
    else fail 'Not ready. Some required tools are missing (see the list above).'; fi
    echo '  Fix the error, then run this script again. It skips what is already installed:'
    echo "  $RUN_COMMAND"
    exit 1
fi
[ -n "$FAILED_OPTIONAL" ] && warn "Optional steps did not finish: $FAILED_OPTIONAL"
if [ "$SIGNED_IN" = 1 ]; then ok 'Ready for the workshop.'
else warn 'Almost ready. Sign in to Codex: run this script again, or run: codex login'; fi
echo '  Open a NEW Terminal window so the new tools are found. To open the workshop project, run:'
echo '  code ~/agent-factory'
echo '  To check again later, run:'
echo "  $CHECK_COMMAND"
exit 0
