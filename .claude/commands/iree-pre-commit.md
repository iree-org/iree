---
allowed-tools: Bash(pre-commit run:*), Bash(git add:*), Bash(git status:*), Bash(git diff:*), Bash(bash -c:*)
description: Run pre-commit on modified files and fix issues
---

## Context

**Modified files (staged or unstaged):**
!`git diff --name-only HEAD`

**Pre-commit results on modified files:**
!`bash -c 'files=$(git diff --name-only HEAD); if [ -n "$files" ]; then pre-commit run --files $files --show-diff-on-failure 2>&1; else echo "No modified files to check"; fi'`

**Current git status:**
!`git status --short`

## Your Task

Analyze the pre-commit results above and fix all issues that were **not** auto-fixed:

1. **Auto-fixed issues (IGNORE these)**: When you see "files were modified by this hook", those files were already fixed automatically (e.g., Black formatting, clang-format). No action needed.

2. **Manual fixes needed (FIX these)**: When you see detailed error messages with line numbers and specific issues (e.g., Ruff linting errors, type annotation issues), these require manual fixes.

For each issue requiring manual fixes:
- Read the affected files
- Apply the fixes following the error messages
- Verify the fix with `pre-commit run --files <file>` after editing

After fixing all issues, run `pre-commit run` again to verify everything passes.

**Important**: Only fix files that have real errors. Don't touch files that were auto-formatted unless they still have errors after auto-formatting.
