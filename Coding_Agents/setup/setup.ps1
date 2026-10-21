<#
Coding Agents workshop setup for Windows 10 and 11.

Installs: git, uv, Python 3.12 (through uv), Node.js LTS, the Codex CLI, and
ruff. Then it sets up Codex: workshop defaults, workshop skills, and sign-in.
Safe to run more than one time: it skips tools and settings that are already
there.

Usage (from a normal PowerShell window, not as Administrator):
  powershell -ExecutionPolicy Bypass -File setup.ps1            install everything
  powershell -ExecutionPolicy Bypass -File setup.ps1 -Check     only show what is installed
  powershell -ExecutionPolicy Bypass -File setup.ps1 -Yes       do not ask
#>
param(
    [switch]$Check,
    [switch]$Yes
)

$ErrorActionPreference = 'Continue'
$PythonVersion = '3.12'
$CodexInstaller = 'https://chatgpt.com/codex/install.ps1'
# Empty if the script is not run from a file.
$SkillsSrc = if ($PSScriptRoot) { Join-Path $PSScriptRoot '..\skills' } else { '' }
$SkillsDest = Join-Path $HOME '.agents\skills'
# Codex reads its config from CODEX_HOME when it is set.
$CodexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME '.codex' }
$CodexConfig = Join-Path $CodexHome 'config.toml'

# Theme: MAIC purple, blue, and black. Warnings stay yellow and errors red.
function Say($msg)  { Write-Host "==> $msg" -ForegroundColor Magenta }
function Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "  [!]  $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "  [X]  $msg" -ForegroundColor Red }
function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function Confirm-Step($question) {
    if ($Yes) { return $true }
    $reply = Read-Host "  $question [Y/n]"
    return -not ($reply -match '^[nN]')
}

function Update-SessionPath {
    # winget and the uv installer change the saved PATH in the registry. Put new
    # saved folders first so this window finds the new tools, and keep the
    # folders this window already has (for example from a PowerShell profile).
    $saved = @(
        [Environment]::GetEnvironmentVariable('Path', 'Machine'),
        [Environment]::GetEnvironmentVariable('Path', 'User')
    ) -join ';'
    $current = @($env:Path -split ';' | Where-Object { $_ })
    $new = @($saved -split ';' | Where-Object { $_ -and ($current -notcontains $_) })
    $env:Path = (@($new) + $current) -join ';'
}

function Install-Winget($id, $label) {
    if (-not (Have 'winget')) {
        Fail 'winget was not found. Install "App Installer" from the Microsoft Store, then run this again.'
        return $false
    }
    Say "  Installing $label with winget"
    # Start-Process gives winget the real console. Piped output prints each
    # progress spinner frame on a new line.
    $process = Start-Process -FilePath 'winget' -NoNewWindow -Wait -PassThru -ArgumentList `
        'install', '--exact', '--id', $id, '--silent', '--accept-package-agreements', '--accept-source-agreements'
    # Exit codes are not reliable (a reboot-needed code still means installed),
    # so the caller checks if the command is found. This only shows the code.
    $code = $process.ExitCode
    Update-SessionPath
    if ($code -ne 0) { Warn "winget exit code $code" }
}

function Exit-Script($code) {
    # Right-click > "Run with PowerShell" closes the window when the script ends.
    # Its command line sets the Process policy to Bypass before it runs the
    # script; only then wait, so the student can read the result.
    try {
        $commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId=$PID" -ErrorAction Stop).CommandLine
        if (-not $Yes -and $commandLine -match '(?i)Set-ExecutionPolicy -Scope Process Bypass') {
            Read-Host '  Press Enter to close this window' | Out-Null
        }
    } catch { }
    if ($null -ne $script:OldBackground) { try { $Host.UI.RawUI.BackgroundColor = $script:OldBackground } catch { } }
    exit $code
}

# The MAIC logo: a network-shaped brain. Columns go blue -> cyan -> purple,
# like the club logo. Single quotes keep the backslashes as they are.
$Logo = @'
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
'@

function Show-Logo {
    # Black background for this window while the script runs. Exit-Script
    # puts the old color back.
    try {
        $script:OldBackground = $Host.UI.RawUI.BackgroundColor
        $Host.UI.RawUI.BackgroundColor = 'Black'
        Clear-Host
    } catch { }
    $colors = @('Blue', 'Blue', 'Cyan', 'Cyan', 'White', 'Magenta', 'Magenta', 'DarkMagenta', 'DarkMagenta')
    Write-Host ''
    foreach ($line in ($Logo -split "`r?`n")) {
        Write-Host '  ' -NoNewline
        for ($i = 0; $i -lt $line.Length; $i++) {
            $color = $colors[[Math]::Min([int][Math]::Floor($i / 5), $colors.Count - 1)]
            Write-Host $line[$i] -NoNewline -ForegroundColor $color
        }
        Write-Host ''
    }
    Write-Host ''
    Write-Host '   MSOE AI Club' -NoNewline -ForegroundColor Cyan
    Write-Host '  |  ' -NoNewline -ForegroundColor DarkGray
    Write-Host 'Coding Agents Workshop setup' -ForegroundColor Magenta
    Write-Host ''
}

function First-Line($text) { if ($text) { ("$text" -split "`n")[0].Trim() } else { '' } }

# npm, npx, and an npm-installed codex are .ps1 files. A new PowerShell window
# runs them only if the execution policy allows local scripts. This script runs
# with Bypass, so it must check the policy that a new window gets.
$ScriptPolicyOk = @('RemoteSigned', 'Unrestricted', 'Bypass')
function Get-NewWindowPolicy {
    foreach ($scope in 'MachinePolicy', 'UserPolicy', 'CurrentUser', 'LocalMachine') {
        $policy = "$(Get-ExecutionPolicy -Scope $scope)"
        if ($policy -ne 'Undefined') { return @{ Policy = $policy; Scope = $scope } }
    }
    # Nothing set: Windows 10 and 11 use Restricted.
    return @{ Policy = 'Restricted'; Scope = 'Default' }
}

function Get-PythonCommand {
    # The 'python' a new window runs. The Microsoft Store alias prints
    # "Python was not found" and exits with an error, so it does not count.
    if (-not (Have 'python')) { return '' }
    $out = python --version 2>&1
    if ($LASTEXITCODE -eq 0 -and "$out" -match '^Python 3') { return "$out".Trim() }
    return ''
}

$script:FailedSteps = @()
$script:FailedOptional = @()
function Run-Step($label, [scriptblock]$body, [switch]$Optional) {
    Say $label
    $result = $false
    try {
        # A PowerShell block returns all of its output, not only the return value.
        # Show tool output to the user and keep only the step's true/false result.
        $result = & $body |
            ForEach-Object { if ($_ -is [bool]) { $_ } else { Write-Host "  $_" } } |
            Select-Object -Last 1
    } catch { Fail $_.Exception.Message }
    if (-not $result) {
        if ($Optional) { Warn "$label did not finish (optional)." }
        else { Fail "$label did not finish. See the messages above." }
        if ($Optional) { $script:FailedOptional += $label } else { $script:FailedSteps += $label }
    }
}

# ---------- version report ----------

function Report-Row($label, $version, $required) {
    if ($version) {
        Write-Host ("  [OK] {0,-16} {1}" -f $label, $version) -ForegroundColor Cyan
    } elseif ($required) {
        Write-Host ("  [X]  {0,-16} missing" -f $label) -ForegroundColor Red
        $script:MissingRequired = $true
    } else {
        Write-Host ("  [-]  {0,-16} missing (optional)" -f $label) -ForegroundColor Yellow
    }
}

function Show-Report {
    $script:MissingRequired = $false
    # Only the exit code of 'uv python find' is used. Its path output can be
    # garbled in Windows PowerShell 5.1 when the user name has letters like e-acute.
    $py = ''
    if (Have 'uv') { uv python find $PythonVersion *> $null; if ($LASTEXITCODE -eq 0) { $py = 'installed' } }
    $policy = Get-NewWindowPolicy
    Say 'Workshop tools'
    Report-Row 'git'            $(if (Have 'git')  { (git --version) -replace 'git version ', '' }) $true
    Report-Row 'uv'             $(if (Have 'uv')   { (uv --version) -replace 'uv ', '' }) $true
    Report-Row "Python $PythonVersion" $py $true
    Report-Row 'python command' (Get-PythonCommand) $true
    Report-Row 'node'           $(if (Have 'node') { node --version }) $true
    Report-Row 'npm'            $(if (Have 'npm')  { npm --version }) $true
    $policyText = ''
    if ($ScriptPolicyOk -contains $policy.Policy) { $policyText = $policy.Policy }
    elseif ($policy.Scope -like '*Policy') { $policyText = "$($policy.Policy), set by your organization: use npm.cmd, npx.cmd" }
    Report-Row 'script policy'  $policyText $true
    Report-Row 'codex'          $(if (Have 'codex') { First-Line (codex --version) }) $true
    Report-Row 'ruff'           $(if (Have 'ruff') { ruff --version }) $false
    $script:SignedIn = $false
    if (Have 'codex') { codex login status *> $null; $script:SignedIn = ($LASTEXITCODE -eq 0) }
    Report-Row 'Codex sign-in'  $(if ($script:SignedIn) { 'yes' }) $false

    if (Have 'git') {
        $name = git config --global user.name
        $email = git config --global user.email
        if ($name -and $email) { Ok "git identity: $name <$email>" }
        else { Warn 'git identity is not set (git config --global user.name / user.email)' }
    }
}

Update-SessionPath

$ThisScript = if ($PSCommandPath) { $PSCommandPath } else { 'setup.ps1' }
$RunCommand = "powershell -ExecutionPolicy Bypass -File `"$ThisScript`""
$CheckCommand = "$RunCommand -Check"

Show-Logo

if ($Check) {
    Show-Report
    Write-Host ''
    if ($script:MissingRequired) {
        Fail 'Some required tools are missing. To install them, run:'
        Write-Host "  $RunCommand"
        Exit-Script 1
    }
    if (-not $script:SignedIn) { Warn 'Codex is not signed in. Run: codex login' }
    else { Ok 'Ready for the workshop.' }
    Exit-Script 0
}

# ---------- install steps ----------

Say "Coding Agents workshop setup (Windows, $env:PROCESSOR_ARCHITECTURE)"
Write-Host "  This installs: git, uv, Python $PythonVersion, Node.js LTS, Codex CLI, ruff."
Write-Host '  Tools you already have are skipped.'

try {
    Invoke-WebRequest -Uri 'https://astral.sh' -UseBasicParsing -TimeoutSec 10 | Out-Null
} catch {
    Warn 'Could not reach astral.sh (offline, or the network blocks it). The uv and Python steps can fail. Other steps continue.'
}
if (-not (Confirm-Step 'Continue?')) { Exit-Script 0 }

Run-Step 'git' {
    # Agents use git to see what they changed (the review skill runs git diff).
    if (-not (Have 'git')) {
        Install-Winget 'Git.Git' 'git'
    }
    if (-not (Have 'git')) { return $false }
    Ok (git --version)

    # git needs a name and email before it can save a commit.
    $name = git config --global user.name
    $email = git config --global user.email
    if ($name -and $email) { Ok "git identity: $name <$email>"; return $true }
    if ($Yes) {
        Warn 'git identity is not set. Run: git config --global user.name "Your Name"'
        Warn 'and: git config --global user.email "you@example.com"'
        return $true
    }
    Write-Host '  git needs a name and email to save commits.'
    if (-not $name) {
        for ($i = 0; $i -lt 3 -and -not $name; $i++) { $name = "$(Read-Host '  Your name')".Trim() }
        if ($name) { git config --global user.name $name }
    }
    if (-not $email) {
        for ($i = 0; $i -lt 3 -and -not $email; $i++) {
            $answer = "$(Read-Host '  Your email')".Trim()
            if ($answer -match '^[^@\s]+@[^@\s]+$') { $email = $answer }
            else { Warn 'That is not an email address (example: name@example.com).' }
        }
        if ($email) { git config --global user.email $email }
    }
    if (-not ($name -and $email)) {
        Warn 'git identity is not set. Run: git config --global user.name "Your Name"'
        Warn 'and: git config --global user.email "you@example.com"'
    }
    return $true
}

Run-Step 'PowerShell script policy' {
    $policy = Get-NewWindowPolicy
    if ($ScriptPolicyOk -contains $policy.Policy) { Ok "Script policy is $($policy.Policy)"; return $true }
    if ($policy.Scope -like '*Policy') {
        Warn "Your organization sets the script policy to $($policy.Policy). You cannot change it."
        Warn 'In PowerShell, type npm.cmd, npx.cmd, and codex.cmd (if codex came from npm) instead.'
        return $true
    }
    Write-Host "  New PowerShell windows block scripts (policy: $($policy.Policy)). npm and npx need them."
    if (-not (Confirm-Step 'Allow local scripts for your user only (RemoteSigned)? No admin needed.')) {
        Warn 'Not changed. npm and npx will not work in a new PowerShell window. Run this script again and answer Y.'
        return $false
    }
    # Windows PowerShell can show an error here because this run uses Bypass.
    # The change still applies, so check the result instead of the error.
    try { Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force -ErrorAction Stop } catch { }
    $policy = Get-NewWindowPolicy
    if ($ScriptPolicyOk -contains $policy.Policy) { Ok "Script policy is $($policy.Policy)"; return $true }
    return $false
}

Run-Step 'uv' {
    if (Have 'uv') { Ok "uv $((uv --version) -replace 'uv ', '')"; return $true }
    powershell -NoProfile -ExecutionPolicy Bypass -Command 'irm https://astral.sh/uv/install.ps1 | iex'
    Update-SessionPath
    return (Have 'uv')
}

Run-Step "Python $PythonVersion" {
    if (-not (Have 'uv')) { return $false }
    $version = Get-PythonCommand
    if ($version -like "Python $PythonVersion.*") { Ok "python command: $version"; return $true }
    # --default also adds 'python' and 'python3' commands to uv's bin folder.
    uv python install $PythonVersion --default
    if ($LASTEXITCODE -ne 0) {
        # Older uv versions need --preview for --default.
        uv python install $PythonVersion --default --preview
        if ($LASTEXITCODE -ne 0) { return $false }
    }
    # Puts uv's Python folder on the user PATH if it is not there (for example
    # when uv came from winget). Older uv versions do not have this command.
    uv python update-shell *> $null
    Update-SessionPath
    $version = Get-PythonCommand
    if (-not $version) {
        Fail '"python" opens the Microsoft Store. Turn off the python.exe and python3.exe app execution aliases in Settings > Apps > Advanced app settings.'
        return $false
    }
    Ok "python command: $version"
    return $true
}

Run-Step 'Node.js' {
    if (Have 'node') { Ok "node $(node --version)"; return $true }
    Install-Winget 'OpenJS.NodeJS.LTS' 'Node.js LTS'
    return (Have 'node')
}

Run-Step 'Codex CLI' {
    if (Have 'codex') { Ok (First-Line (codex --version)); return $true }
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm $CodexInstaller | iex"
    Update-SessionPath
    if (Have 'codex') { return $true }
    # Fallback: the npm package works anywhere Node.js is installed.
    if (Have 'npm') {
        Warn 'The Codex installer did not finish. Trying npm instead.'
        npm install -g @openai/codex
        Update-SessionPath
    }
    return (Have 'codex')
}

Run-Step 'ruff' -Optional {
    if (Have 'ruff') { Ok (ruff --version); return $true }
    if (-not (Have 'uv')) { return $false }
    uv tool install ruff *> $null
    Update-SessionPath
    if (-not (Have 'ruff')) { return $false }
    Ok (ruff --version)
    return $true
}

Run-Step 'Codex settings' {
    # Workshop defaults: Codex can edit files in the project and use the network
    # (npm install, uv sync), and asks before it does anything outside it.
    # Without [windows] sandbox, Codex runs with no sandbox on Windows;
    # "unelevated" needs no admin. An existing config is never changed.
    if (Test-Path $CodexConfig) {
        Ok "$CodexConfig already exists, not changed"
        if (-not (Select-String -Path $CodexConfig -Pattern '^\s*(\[windows\]|windows\.sandbox\s*=)' -Quiet)) {
            Warn 'It has no Windows sandbox setting, so Codex runs without a sandbox. Add these lines to it:'
            Write-Host '      [windows]'
            Write-Host '      sandbox = "unelevated"'
        }
    } else {
        New-Item -ItemType Directory -Force -Path (Split-Path $CodexConfig) | Out-Null
        $config = @(
            '# Made by the AI Club workshop setup script. Edit it as you like.',
            'approval_policy = "on-request"',
            'sandbox_mode = "workspace-write"',
            '',
            '[sandbox_workspace_write]',
            'network_access = true',
            '',
            '[windows]',
            'sandbox = "unelevated"'
        )
        # Write without a byte order mark: Windows PowerShell 5.1 adds one to UTF8 files.
        [IO.File]::WriteAllLines($CodexConfig, $config)
        Ok "Made $CodexConfig"
    }

    if (-not $SkillsSrc -or -not (Test-Path $SkillsSrc)) {
        Warn "Workshop skills not copied: no skills folder at $SkillsSrc"
        Warn 'The setup and skills folders must be next to each other. Run setup.ps1 from the extracted folder.'
        return $true
    }
    New-Item -ItemType Directory -Force -Path $SkillsDest | Out-Null
    $copyFailed = $false
    foreach ($skill in Get-ChildItem -Path $SkillsSrc -Directory) {
        $source = Join-Path $skill.FullName 'SKILL.md'
        if (-not (Test-Path $source)) { continue }
        $target = Join-Path $SkillsDest $skill.Name
        # SKILL.md is copied last, so a skill that has it is complete. A folder
        # without it is from a copy that did not finish, so it is copied again.
        if (Test-Path (Join-Path $target 'SKILL.md')) { Ok "Skill $($skill.Name) already exists, not changed"; continue }
        try {
            New-Item -ItemType Directory -Force -Path $target | Out-Null
            Get-ChildItem -Path $skill.FullName -Exclude 'SKILL.md' |
                Copy-Item -Destination $target -Recurse -Force -ErrorAction Stop
            Copy-Item -Path $source -Destination $target -Force -ErrorAction Stop
            Ok "Copied skill $($skill.Name)"
        } catch {
            Fail "Could not copy skill $($skill.Name): $($_.Exception.Message)"
            $copyFailed = $true
        }
    }
    return (-not $copyFailed)
}

Run-Step 'Codex sign-in' -Optional {
    if (-not (Have 'codex')) { return $false }
    codex login status *> $null
    if ($LASTEXITCODE -eq 0) { Ok 'Signed in to Codex'; return $true }
    if ($Yes -or -not (Confirm-Step 'Sign in to Codex now? Use your ChatGPT account.')) {
        Warn 'Not signed in. Later, run: codex login'
        return $true
    }
    # Sign-in runs in its own window, so closing it (or Ctrl+C there) does not
    # stop this script.
    Write-Host '  A new window opens for sign-in, then your browser. If it gets stuck, close that window.'
    # WaitForExit waits for this window only. Start-Process -Wait would also wait
    # for the browser if codex started it. If sign-in fails, the window stays
    # open so the student can read the error.
    # -EncodedCommand keeps the quotes; Start-Process does not quote arguments.
    $child = 'codex login; if ($LASTEXITCODE -ne 0) { Read-Host "Sign-in did not finish. Press Enter to close" }'
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($child))
    $process = Start-Process -FilePath 'powershell' -PassThru `
        -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', $encoded
    $process.WaitForExit()
    codex login status *> $null
    if ($LASTEXITCODE -eq 0) { Ok 'Signed in to Codex'; return $true }
    Warn 'Not signed in. Later, run: codex login'
    return $false
}

Write-Host ''
Show-Report

Write-Host ''
if ($script:FailedSteps.Count -gt 0 -or $script:MissingRequired) {
    if ($script:FailedSteps.Count -gt 0) {
        Fail ('Not ready. These steps did not finish: ' + ($script:FailedSteps -join ', '))
    } else {
        Fail 'Not ready. Some required tools are missing (see the list above).'
    }
    Write-Host '  Fix the error, then run this script again. It skips what is already installed:'
    Write-Host "  $RunCommand"
    Exit-Script 1
}
if ($script:FailedOptional.Count -gt 0) {
    Warn ('Optional steps did not finish: ' + ($script:FailedOptional -join ', '))
}
if (-not $script:SignedIn) {
    Warn 'Almost ready. Sign in to Codex: run "codex login" in a new PowerShell window.'
} else {
    Ok 'Ready for the workshop.'
}
Write-Host '  Open a NEW PowerShell window so the new tools are found. To check again later, run:'
Write-Host "  $CheckCommand"
Exit-Script 0
