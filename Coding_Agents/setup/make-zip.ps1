<#
Builds the zip file that students download from Teams, in Coding_Agents\dist.

Usage:
  powershell -ExecutionPolicy Bypass -File make-zip.ps1 -KeyFile C:\path\to\key.txt
  powershell -ExecutionPolicy Bypass -File make-zip.ps1
      Without -KeyFile, the zip has no key and the setup scripts ask students to paste it.

The key file holds one line: the workshop project's API key (sk-...).
#>
param([string]$KeyFile)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot
$dist = Join-Path $root 'dist'
$stage = Join-Path $dist 'coding-agents-workshop'
$zip = Join-Path $dist 'coding-agents-workshop.zip'

if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path (Join-Path $stage 'setup') | Out-Null

Copy-Item (Join-Path $PSScriptRoot 'setup.ps1'), (Join-Path $PSScriptRoot 'README.md') (Join-Path $stage 'setup')
# bash on macOS fails on Windows line endings, whatever git did to this checkout.
$sh = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'setup.sh')).Replace("`r`n", "`n")
[IO.File]::WriteAllText((Join-Path $stage 'setup\setup.sh'), $sh)
Copy-Item (Join-Path $root 'coding-agents-workshop.md') $stage
foreach ($folder in 'skills', 'starter') {
    # Leaves out files made by running the project locally.
    robocopy (Join-Path $root $folder) (Join-Path $stage $folder) /E /NFL /NDL /NJH /NJS /NP `
        /XD .venv __pycache__ .pytest_cache .ruff_cache .git /XF uv.lock | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Could not copy $folder (robocopy exit code $LASTEXITCODE)" }
}

if ($KeyFile) {
    $key = "$(Get-Content -Path $KeyFile -Raw)".Trim()
    if ($key -notmatch '^sk-\S+$') { throw "$KeyFile does not hold an API key (it should start with sk-)." }
    [IO.File]::WriteAllText((Join-Path $stage 'workshop-key.txt'), $key)
} else {
    Write-Host 'No -KeyFile: the zip has no key. The setup scripts will ask students to paste it.' -ForegroundColor Yellow
}

if (Test-Path $zip) { Remove-Item -Force $zip }
# Windows PowerShell writes backslashes in entry names, which Mac unzip keeps as
# part of the file name, so name each entry with forward slashes.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::Open($zip, 'Create')
try {
    foreach ($file in Get-ChildItem -Path $stage -Recurse -File -Force) {
        $name = $file.FullName.Substring($stage.Length + 1).Replace('\', '/')
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $file.FullName, $name) | Out-Null
    }
} finally {
    $archive.Dispose()
}
Remove-Item -Recurse -Force $stage
Write-Host "Made $zip" -ForegroundColor Cyan
