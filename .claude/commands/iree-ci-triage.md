---
allowed-tools: Bash(iree-ci-triage:*), Bash(gh:*), Read, Grep
description: Triage CI failures for a PR or commit
---

## Context

**Target:**
!`if [ -n "$ARGUMENTS" ]; then echo "PR/run: $ARGUMENTS"; else echo "No PR specified - showing recent failing PRs"; fi`

**CI Triage Results:**
!`if [ -n "$ARGUMENTS" ]; then PYTHONPATH=tools/utils python3 -m ci.iree_ci_triage --pr "$ARGUMENTS" --checklist 2>&1 | head -200; else gh pr list --state open --limit 10 --json number,title,headRefName --template '{{range .}}#{{.number}} {{.title}} ({{.headRefName}}){{"\n"}}{{end}}' 2>&1; fi`

## Your Task

Analyze the CI triage results above and help fix the failures:

1. **If no PR was specified**: Ask which PR to triage from the list shown

2. **For each failure category**:
   - **Compile errors**: Read the failing file and fix the issue
   - **Test failures**: Run `iree-lit-test` to debug, then fix CHECK patterns or code
   - **Sanitizer issues**: Analyze the stack trace and fix memory/thread issues
   - **Infrastructure flakes**: Note these are transient, suggest re-running

3. **Priority order**: Fix errors before warnings, compile errors before test failures

4. **After fixing**: Verify the fix locally before suggesting it's ready

Use `iree-ci-triage --pr N --json` for machine-readable output if needed.
Use `iree-lit-lint` and `iree-lit-test` for test-related failures and reference the `iree-lit-tools` skill for guidance on style and common workflows.
