# IREE CI Tools

Quick start guide for IREE's CI health management tools.

## Available Tools

### iree-ci-triage
Analyzes CI failures and identifies root causes from GitHub Actions runs.

**Quick Start:**
```bash
# Triage a specific run
iree-ci-triage --run 12345678

# Triage latest failure from PR
iree-ci-triage --pr 22625

# JSON output for automation
iree-ci-triage --run 12345678 --json
```

**[Full documentation →](iree_ci_triage.md)**

### iree-ci-garden
Manages a corpus of CI failures for pattern development and health tracking. Tracks commit hashes, PR numbers, re-runs, and recognition rates.

**Quick Start:**
```bash
# Fetch failures (last 24 hours by default)
iree-ci-garden fetch

# Classify logs using iree-ci-triage
iree-ci-garden classify

# Check corpus health metrics
iree-ci-garden status

# Search for specific error patterns
iree-ci-garden search "undefined reference"
```

**[Full documentation →](iree_ci_garden.md)**

## Installation

These tools are part of IREE's Python utilities:
```bash
cd tools/utils
pip install -e .
```

## Common Workflows

### Daily CI Health Check
```bash
# 1. Fetch overnight failures (includes commit hashes, PR numbers, re-run attempts)
iree-ci-garden fetch

# 2. Classify new failures
iree-ci-garden classify

# 3. Review corpus health and unrecognized patterns
iree-ci-garden status
cat ~/.iree-ci-corpus/unrecognized/TODO.md

# 4. Investigate specific failures
iree-ci-triage --run <run_id>
```

### Investigating a PR Failure
```bash
# Quick triage of PR
iree-ci-triage --pr 22625

# Get detailed breakdown
iree-ci-triage --pr 22625 --checklist

# Add PR failures to corpus for tracking
iree-ci-garden fetch --pr 22625

# Find all failures from this PR
cat ~/.iree-ci-corpus/corpus.jsonl | jq 'select(.pr_number == 22625)'
```

### Analyzing Flaky Tests
```bash
# Fetch recent failures
iree-ci-garden fetch --since 2025-11-01

# Find tests that passed on retry (attempt > 1)
cat ~/.iree-ci-corpus/corpus.jsonl | jq 'select(.attempt > 1)'

# Check which jobs are flaky
cat ~/.iree-ci-corpus/corpus.jsonl | jq 'select(.attempt > 1) | {url, attempt, workflow}'
```

### Commit-Based Failure Analysis
```bash
# Fetch failures from last week
iree-ci-garden fetch --since 2025-11-10

# Find all failures for specific commit
cat ~/.iree-ci-corpus/corpus.jsonl | jq 'select(.head_sha == "abc123...")'

# Track failures by branch
cat ~/.iree-ci-corpus/corpus.jsonl | jq 'select(.branch == "main") | .head_sha' | sort | uniq -c
```

### Testing New Error Patterns
```bash
# 1. Add pattern to patterns.yaml
vim tools/utils/ci/patterns.yaml

# 2. Test against entire corpus
iree-ci-garden classify --force

# 3. Check recognition rate improvement
iree-ci-garden status

# 4. Review remaining unrecognized failures
cat ~/.iree-ci-corpus/unrecognized/TODO.md
```

## Configuration

Both tools use GitHub CLI (`gh`) for API access:
```bash
# Check authentication
gh auth status

# Login if needed
gh auth login
```

## Environment Variables

- `IREE_CI_CORPUS_DIR` - CI garden corpus location (default: `~/.iree-ci-corpus`)
- `GH_TOKEN` - GitHub authentication token (usually managed by `gh auth`)

## Getting Help

For detailed usage and options:
```bash
iree-ci-triage --help
iree-ci-garden --help
```

For troubleshooting and development:
- [iree-ci-triage documentation](iree_ci_triage.md)
- [iree-ci-garden documentation](iree_ci_garden.md)
