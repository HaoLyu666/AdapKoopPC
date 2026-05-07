# GitHub Setup

This machine is prepared with:

- Git for Windows
- GitHub CLI (`gh`) installed at `C:\Program Files\GitHub CLI\gh.exe`
- The local repository initialized on branch `main`
- All project files staged for the initial commit

## Publish From PowerShell

Open PowerShell in this folder:

```powershell
cd "D:\博\科研\文章\2024TRB\混合车队\code\mix_platoon\AdapKoopPC"
```

Run the helper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\github_publish.ps1 -RepoName AdapKoopPC -Visibility public
```

The helper will:

1. Ask for your Git commit name and email if they are not already configured.
2. Open GitHub login through `gh auth login`.
3. Create the GitHub repository if no remote is configured.
4. Push branch `main` to GitHub.

Use `-Visibility private` if you want to upload privately first.

## Manual Commands

If you prefer to do it manually:

```powershell
git config user.name "Your Name"
git config user.email "your_email@example.com"
gh auth login --web --git-protocol https
git commit -m "Initial open-source release"
gh repo create AdapKoopPC --public --source . --remote origin --push
```

If the repository already exists on GitHub:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/AdapKoopPC.git
git push -u origin main
```

## If `AdapKoopPC` Already Exists as a Fork

Force-pushing new code to an existing fork does **not** remove the "forked from ..." label. That label is GitHub repository metadata, independent from Git history. Use one of these approaches first.

### Option A: Leave the Fork Network

Use this if GitHub shows the option.

1. Open the existing fork on GitHub.
2. Go to **Settings** -> **General** -> **Danger Zone**.
3. Click **Leave fork network**.
4. After it becomes a standalone repository, run:

```powershell
git config user.name "Your Name"
git config user.email "your_email@example.com"
git commit -m "Initial open-source release"
git remote add origin https://github.com/YOUR_USERNAME/AdapKoopPC.git
git push -u origin main --force
```

### Option B: Rename or Delete the Fork, Then Create a Fresh Repository

Use this if the **Leave fork network** option is unavailable, or if you want a completely fresh repository.

1. Open the existing fork on GitHub.
2. Either rename it to something like `AdapKoopPC-fork-backup`, or delete it if you do not need it.
3. Back in this local folder, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\github_publish.ps1 -RepoName AdapKoopPC -Visibility public
```

Renaming is safer than deletion because deletion permanently removes repository metadata such as issues, pull requests, settings, and stars.
