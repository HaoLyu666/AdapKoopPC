param(
    [string]$RepoName = "AdapKoopPC",
    [ValidateSet("public", "private", "internal")]
    [string]$Visibility = "public",
    [string]$UserName = "",
    [string]$UserEmail = ""
)

$ErrorActionPreference = "Stop"

function Find-Gh {
    $cmd = Get-Command gh -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        "$env:ProgramFiles\GitHub CLI\gh.exe",
        "$env:LOCALAPPDATA\Programs\GitHub CLI\gh.exe",
        "$env:ProgramFiles(x86)\GitHub CLI\gh.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    throw "GitHub CLI was not found. Install it with: winget install --id GitHub.cli -e --source winget"
}

function Has-Commit {
    git rev-parse --verify HEAD *> $null
    return $LASTEXITCODE -eq 0
}

$projectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

$gh = Find-Gh
Write-Host "Using GitHub CLI: $gh"

if (-not $UserName) {
    $UserName = git config user.name
}
if (-not $UserName) {
    $UserName = Read-Host "Git commit name"
}
if (-not $UserEmail) {
    $UserEmail = git config user.email
}
if (-not $UserEmail) {
    $UserEmail = Read-Host "Git commit email"
}

git config user.name $UserName
git config user.email $UserEmail
git config core.autocrlf false
git config core.eol lf

& $gh auth status
if ($LASTEXITCODE -ne 0) {
    & $gh auth login --web --git-protocol https
}

git add .
if (-not (Has-Commit)) {
    git commit -m "Initial open-source release"
}
else {
    $pending = git status --porcelain
    if ($pending) {
        git commit -m "Update open-source package"
    }
}

$remote = git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0 -or -not $remote) {
    $visibilityFlag = "--$Visibility"
    & $gh repo create $RepoName $visibilityFlag --source . --remote origin --push
}
else {
    git push -u origin main
}

Write-Host "Done."
