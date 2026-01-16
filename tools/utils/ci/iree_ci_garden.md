# iree-ci-garden - CI Corpus Management Tool

`iree-ci-garden` manages a corpus of CI failures for pattern development and health tracking. It fetches failures from GitHub Actions, classifies them using `iree-ci-triage`, tracks recognition rates, and generates TODO lists for unrecognized patterns.

## Overview

The tool separates data collection (fetch) from analysis (classify) to enable iterative improvement of error pattern detection:

1. **Fetch** - Download CI failures and logs from GitHub Actions
2. **Classify** - Run `iree-ci-triage` to identify root causes
3. **Track** - Monitor recognition rates and pattern coverage
4. **Garden** - Identify unrecognized failures needing new patterns

**Goal**: Achieve 100% recognition rate for CI failures through continuous pattern development.

## Installation

```bash
# Install IREE CI tools
cd tools/utils
pip install -e .

# Verify installation
iree-ci-garden --help

# Authenticate with GitHub (required)
gh auth login
```

## Quick Start

```bash
# Daily workflow: fetch and classify overnight failures
iree-ci-garden fetch                 # Fetch last 24 hours
iree-ci-garden classify              # Run classification
iree-ci-garden status                # Check health metrics

# Check TODO list for unrecognized patterns
cat ~/.iree-ci-corpus/unrecognized/TODO.md
```

## Commands

### fetch - Fetch CI Failures

Downloads workflow run failures from GitHub Actions and stores failed job logs in the corpus.

```bash
# Fetch failures from last 24 hours (default)
iree-ci-garden fetch

# Fetch from specific date range
iree-ci-garden fetch --since 2025-11-01

# Fetch from specific PR
iree-ci-garden fetch --pr 22625

# Fetch specific run
iree-ci-garden fetch --run 12345678

# Fetch from specific branch
iree-ci-garden fetch --branch staging

# Quiet mode (no progress output)
iree-ci-garden fetch --quiet

# Limit number of runs
iree-ci-garden fetch --limit 50
```

**Captured Metadata** (v2.0 schema):
- Run info: `run_id`, `workflow`, `name`, `url`
- Git info: `head_sha`, `branch`, `pr_number`
- Trigger: `event` (push, pull_request, schedule, etc.)
- Status: `status`, `conclusion`, `attempt` (for re-runs)
- Timing: `created_at`, `started_at`, `updated_at`
- Jobs: `jobs` count, `failed_jobs` count

### classify - Classify Failure Logs

Runs `iree-ci-triage` on corpus logs to identify root causes and categories. Results are cached with 30-day TTL.

```bash
# Classify all unclassified logs
iree-ci-garden classify

# Force reclassification (ignores cache)
iree-ci-garden classify --force

# Classify specific run
iree-ci-garden classify --run 12345678

# Quiet mode
iree-ci-garden classify --quiet
```

**Output**:
- Recognition rate (% of logs with identified root causes)
- Failure categories and counts
- Location of unrecognized failure TODO list

**After classification**, check:
```bash
# View TODO list for unrecognized patterns
cat ~/.iree-ci-corpus/unrecognized/TODO.md
```

### status - Show Corpus Health

Displays corpus statistics and health metrics.

```bash
# Human-readable status
iree-ci-garden status

# JSON output for automation
iree-ci-garden status --json

# Quiet mode (exit code only)
iree-ci-garden status --quiet
```

**Metrics**:
- Total runs and logs in corpus
- Total size on disk
- Time span (earliest to latest failure)
- Recognition rate (goal: >90%)
- Top failure categories

### search - Search Corpus Logs

Search all corpus logs for a regex pattern (case-insensitive).

```bash
# Search for error pattern
iree-ci-garden search "undefined reference"

# Search with category filter
iree-ci-garden search "assertion failed" --category compilation

# Search date range
iree-ci-garden search "timeout" --since 2025-11-01

# Quiet mode (no output, exit code only)
iree-ci-garden search "pattern" --quiet
```

**Output**: Shows matching log paths and line numbers for investigation.

## Corpus Structure

The corpus is stored in `~/.iree-ci-corpus` (configurable via `--corpus-dir`):

```
~/.iree-ci-corpus/
├── config.json                    # Corpus configuration (v2.0 schema)
├── corpus.jsonl                   # Main index (streaming JSONL)
├── daily/                         # Daily fetch manifests
│   └── 2025-11-17.json
├── runs/                          # Per-run metadata
│   └── 12345678.json
├── logs/                          # Raw logs organized by run
│   └── 12345678/
│       ├── 55555555.log          # Job logs
│       └── 66666666.log
├── classification/                # Classification results
│   ├── cache/                    # Cached triage results (30-day TTL)
│   │   └── 12345678_55555555.json
│   └── history/                  # Classification history
└── unrecognized/                 # Failures needing attention
    ├── TODO.md          # Generated TODO list
    └── samples/                  # Sample error snippets
```

## Schema Evolution

### v2.0 Schema (Current)

Enhanced metadata capture for comprehensive analysis:

**New fields**:
- `head_sha` - Commit hash (enables code correlation)
- `event` - Trigger type (push, pull_request, schedule, etc.)
- `pr_number` - PR number for PR events
- `url` - Direct GitHub link
- `attempt` - Re-run count (flake detection)
- `status` - Run status (queued, in_progress, completed)
- `started_at`, `updated_at` - Timing info
- `workflow_id` - Workflow database ID

**Analysis capabilities enabled**:
- Link failures to specific commits via `head_sha`
- Track PR-specific failures via `pr_number`
- Detect flaky tests via `attempt` count
- Compare scheduled vs PR failures via `event`
- Direct debugging via `url` links

### Migration from v1.0

Not required - the corpus can be wiped and re-fetched with the new schema:

```bash
# Backup old corpus (optional)
mv ~/.iree-ci-corpus ~/.iree-ci-corpus.v1.0.backup

# Fetch with new schema
iree-ci-garden fetch --since 2025-11-01
```

## Advanced Usage

### Analyzing Flaky Tests

Find failures that passed on retry (attempt > 1):

```bash
# Fetch and classify
iree-ci-garden fetch
iree-ci-garden classify

# Find flaky runs
cat ~/.iree-ci-corpus/corpus.jsonl | jq 'select(.attempt > 1)'
```

### Commit-Based Analysis

Find all failures for a specific commit:

```bash
# Fetch failures
iree-ci-garden fetch --since 2025-11-01

# Find failures for commit
cat ~/.iree-ci-corpus/corpus.jsonl | jq 'select(.head_sha == "abc123...")'
```

### PR Failure Analysis

Track failures by PR:

```bash
# Fetch PR failures
iree-ci-garden fetch --pr 22625

# Find all PR failures
cat ~/.iree-ci-corpus/corpus.jsonl | jq 'select(.event == "pull_request")'
```

### Pattern Development Workflow

1. Classify corpus to identify unrecognized failures:
   ```bash
   iree-ci-garden classify
   cat ~/.iree-ci-corpus/unrecognized/TODO.md
   ```

2. Add new patterns to `patterns.yaml` (see iree-ci-triage docs)

3. Test new patterns against corpus:
   ```bash
   iree-ci-garden classify --force
   iree-ci-garden status  # Check recognition rate
   ```

4. Iterate until recognition rate >90%

### Automation Examples

**Daily cron job**:
```bash
#!/bin/bash
# Daily CI health check
iree-ci-garden fetch --quiet
iree-ci-garden classify --quiet

# Alert if recognition rate drops below 90%
RATE=$(iree-ci-garden status --json | jq -r '.recognition.overall_rate')
if (( $(echo "$RATE < 0.9" | bc -l) )); then
  echo "Warning: Recognition rate dropped to $RATE"
fi
```

**CI integration**:
```yaml
# GitHub Actions workflow
- name: Check CI health
  run: |
    iree-ci-garden fetch --since $(date -d '7 days ago' +%Y-%m-%d)
    iree-ci-garden classify
    iree-ci-garden status --json > ci-health.json
```

## Configuration

### Environment Variables

- `GH_TOKEN` - GitHub authentication token (managed by `gh auth`)

### Corpus Configuration

Edit `~/.iree-ci-corpus/config.json`:

```json
{
  "corpus_version": "2.0",
  "github_repo": "iree-org/iree",
  "fetch_settings": {
    "default_limit": 100,
    "include_branches": ["*"]
  },
  "classification_settings": {
    "iree_ci_triage_args": ["--json", "--no-context"],
    "cache_results": true,
    "cache_ttl_days": 30
  }
}
```

## Troubleshooting

### "GitHub CLI (gh) not found"

Install GitHub CLI:
```bash
# macOS
brew install gh

# Linux
sudo apt install gh  # or equivalent

# Authenticate
gh auth login
```

### "iree-ci-triage not found"

Ensure iree-ci-triage is in PATH:
```bash
# Check installation
which iree-ci-triage

# If not found, install CI tools
cd tools/utils
pip install -e .
```

### Recognition Rate is Low (<50%)

1. Update `iree-ci-triage` patterns (see `patterns.yaml`)
2. Force reclassification: `iree-ci-garden classify --force`
3. Check TODO list: `cat ~/.iree-ci-corpus/unrecognized/TODO.md`
4. Add new patterns for high-priority failures

### Corpus is Too Large

Prune old failures:
```bash
# Keep only last 30 days
find ~/.iree-ci-corpus/logs -mtime +30 -delete
find ~/.iree-ci-corpus/runs -mtime +30 -delete

# Rebuild index
rm ~/.iree-ci-corpus/corpus.jsonl
# Re-index from remaining runs metadata
```

## Development

### Running Tests

```bash
# Unit tests
pytest tools/utils/ci/

# Integration test
iree-ci-garden fetch --limit 5
iree-ci-garden classify
iree-ci-garden status
```

### Adding New Commands

1. Add subparser in `parse_arguments()`
2. Implement `cmd_<name>()` function
3. Add dispatch in `main()`
4. Update help text and documentation

### Code Structure

- `iree_ci_garden.py` - Main CLI with 5 commands
- `core/corpus.py` - Corpus management (JSONL index)
- `core/fetcher.py` - GitHub API integration
- `core/classifier.py` - iree-ci-triage integration
- `core/gardener.py` - Health tracking, TODO generation
- `core/github_client.py` - GitHub CLI wrapper

## See Also

- [iree-ci-triage documentation](iree_ci_triage.md) - Root cause analysis tool
- [CI Tools README](README.md) - Quick start guide
- [GitHub CLI documentation](https://cli.github.com/manual/)
