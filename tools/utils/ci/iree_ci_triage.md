# IREE CI Triage Tools

Automated error detection and triage for IREE CI failures.

---

## Quick Start

```bash
# Triage all failures in a workflow run
iree-ci-triage --run 12345678

# Triage latest failure from a PR
iree-ci-triage --pr 12345

# Triage latest failure from a commit
iree-ci-triage --commit abc123def

# Triage latest failure from a branch
iree-ci-triage --branch main

# Triage specific job
iree-ci-triage --run 12345678 --job 987654321

# JSON output for automation
iree-ci-triage --run 12345678 --json | jq '.summary'
```

---

## Tools

### `iree-ci-triage`

Fetches CI failure logs from GitHub Actions and performs automated triage.

**Features**:
- Detects 28+ error patterns (compile errors, GPU crashes, test failures, etc.)
- Groups co-occurring errors into root causes
- Distinguishes actionable bugs from infrastructure flakes
- Generates fix checklists with file paths and error details
- Supports human-readable markdown and JSON output

**Usage**:
```bash
iree-ci-triage [INPUT] [OPTIONS]

Input (mutually exclusive, one required):
  --run RUN_ID              Workflow run ID
  --pr PR_NUMBER            Pull request number (uses latest failed run)
  --commit SHA              Commit SHA (uses latest failed run)
  --branch BRANCH           Branch name (uses latest failed run)

Options:
  --job JOB_ID              Specific job to triage (default: all failed jobs)
  --repo OWNER/REPO         Repository (default: iree-org/iree)
  --status STATUS           Filter by run status (default: failure)

  --patterns FILE           Custom patterns.yaml
  --rules FILE              Custom cooccurrence_rules.yaml

  --json                    JSON output (to stdout)
  --checklist               Simple checklist for LLMs
  --no-context              Don't include error context in markdown
  --quiet                   Suppress progress messages
```

**Examples**:
```bash
# Human-readable triage by run ID
iree-ci-triage --run 19394282553

# Triage latest failure from a pull request
iree-ci-triage --pr 12345

# Triage latest failure from a specific commit
iree-ci-triage --commit abc123def456

# Triage latest failure from main branch
iree-ci-triage --branch main

# Find successful runs for a PR (for comparison)
iree-ci-triage --pr 12345 --status success

# JSON output for piping
iree-ci-triage --run 19394282553 --json | jq '.jobs[0].root_causes'

# Simple checklist for LLM consumption
iree-ci-triage --pr 12345 --checklist

# Custom error patterns
iree-ci-triage --run 19394282553 --patterns my_patterns.yaml
```

---

## Error Pattern System

### Pattern Definitions (`patterns.yaml`)

Defines error signatures to detect in logs. Each pattern includes:
- **Regex**: Pattern to match error text
- **Severity**: critical, high, medium, or low
- **Actionable**: Whether developers can fix (vs infrastructure flake)
- **Extraction rules**: Pull file paths, error codes, etc.

**Example pattern**:
```yaml
compile_error:
  pattern: 'error: .*|compilation terminated'
  severity: critical
  actionable: true
  context_lines: 5
  description: "C++ compilation error"
  extract:
    - name: file_path
      regex: '(\S+\.\w+):(\d+):(\d+):'
    - name: error_message
      regex: 'error: (.+)'
```

**28 built-in patterns**:
- Build errors: `compile_error`, `linker_error`, `cmake_error`
- Test failures: `test_failed`, `filecheck_failed`, `assertion_failed`
- GPU crashes: `vk_device_lost`, `cuda_device_lost`, `hip_device_lost`
- ROCm issues: `rocclr_memobj` (cleanup crash), `hip_file_not_found`
- Memory errors: `segfault`, `oom`, `stack_overflow`
- Infrastructure: `timeout`, `network_error`, `docker_error`
- ...and more

### Root Cause Rules (`cooccurrence_rules.yaml`)

Defines how co-occurring patterns group into single root causes.

**Example rule**:
```yaml
- name: "rocm_cleanup_crash"
  primary_pattern: "rocclr_memobj"
  secondary_patterns:
    - "aborted"
    - "core_dumped"
    - "process_exit_error"
  description: >
    ROCm memory object cleanup crash. Tests often PASS before
    this crash during teardown. False CI failure.
  priority: 100
  actionable: false
  category: "infrastructure"
```

When a log contains `rocclr_memobj` + `aborted` + `process_exit_error`, they're grouped into one "rocm_cleanup_crash" root cause instead of three separate issues.

---

## Adding New Patterns

See [`PATTERN_GUIDE.md`](PATTERN_GUIDE.md) for detailed instructions.

**Quick steps**:

1. **Find the error signature** in a CI log
2. **Write the pattern**:
   ```yaml
   new_pattern:
     pattern: 'regex to match error'
     severity: high
     actionable: true
     context_lines: 5
     description: "What this error means"
   ```
3. **Add to `patterns.yaml`**
4. **Test it**:
   ```bash
   iree-ci-triage --run <run-with-this-error>
   ```
5. **(Optional) Add co-occurrence rule** if it appears with other patterns

---

## Architecture

```
tools/utils/ci/
├── iree_ci_triage.py        # Main tool entry point
├── patterns.yaml             # Error pattern definitions
├── cooccurrence_rules.yaml   # Root cause grouping rules
├── PATTERN_GUIDE.md          # Pattern authoring guide
├── core/
│   ├── patterns.py           # Pattern loading & matching
│   ├── github_client.py      # GitHub CLI wrapper
│   └── extractors.py         # Output formatters
└── README.md                 # This file

tools/utils/bin/
└── iree-ci-triage            # Executable wrapper
```

**Data flow**:
1. Fetch workflow runs & job logs via GitHub CLI
2. Match error patterns against log content
3. Group co-occurring patterns into root causes
4. Format output (markdown checklist or JSON)

---

## Output Formats

### Human-Readable Markdown

```bash
iree-ci-triage --run 19394282553
```

**Output**:
```markdown
# CI Failure Triage Report

**Run ID**: 19394282553
**Job**: Test Sharktank / rocm_hip_w7900

## Summary

Found **2** root cause(s)
- **1** actionable (code bugs)
- **1** infrastructure (flakes/driver issues)

## Fix Checklist

- [ ] **compilation_failure** (code)
  - C++ compilation error
  - **Severity**: 🔴 critical
  - **Occurrences**: 1 error(s)
  - **Location**: Line 145
  - **Details**:
    - file_path: `/path/to/file.cpp:145:12`
    - error_message: `use of undeclared identifier 'foo'`

## Infrastructure Issues (Non-Actionable)

### rocm_cleanup_crash
- ROCm memory object cleanup crash (false CI failure)
- **Category**: infrastructure
- **Occurrences**: 1 error(s)
```

### JSON Output

```bash
iree-ci-triage --run 19394282553 --json
```

**Output** (to stdout):
```json
{
  "jobs": [
    {
      "run_id": "19394282553",
      "job_id": "52085546287",
      "job_name": "Test Sharktank / rocm_hip_w7900",
      "root_causes": [
        {
          "name": "compilation_failure",
          "category": "code",
          "priority": 80,
          "actionable": true,
          "description": "C++ compilation error...",
          "matches": [...]
        }
      ]
    }
  ],
  "summary": {
    "total_jobs": 1,
    "total_root_causes": 2,
    "actionable_issues": 1
  }
}
```

### Checklist Format (for LLMs)

```bash
iree-ci-triage --run 19394282553 --checklist
```

**Output**:
```
# Fix Checklist for Test Sharktank / rocm_hip_w7900

- [ ] RC-1: compilation_failure
     Line 145: error: use of undeclared identifier 'foo'
     → Fix: File: /path/to/file.cpp, Error: use of undeclared identifier
```

---

## Dependencies

- **GitHub CLI (`gh`)**: Required for fetching workflow data
  - Install: https://cli.github.com
  - Auth: `gh auth login`
- **Python 3.8+**: Standard library + PyYAML

**Check dependencies**:
```bash
gh auth status  # Should show "Logged in to github.com"
```

---

## Integration with Other Tools

### Piping to jq

```bash
# Get all actionable issues
iree-ci-triage --run 12345 --json | jq '.jobs[].root_causes[] | select(.actionable)'

# Count infrastructure failures
iree-ci-triage --run 12345 --json | jq '[.jobs[].root_causes[] | select(.actionable == false)] | length'

# Extract file paths from compile errors
iree-ci-triage --run 12345 --json | jq -r '.jobs[].root_causes[] | select(.name == "compilation_failure") | .matches[].extracted_fields.file_path'
```

### Using with lit_tools

Future integration: Extract LIT test failures and structure them in JSON output automatically.

---

## Extending the System

### Custom Patterns

1. Create `my_patterns.yaml`:
   ```yaml
   patterns:
     my_error:
       pattern: 'MyError: .*'
       severity: high
       actionable: true
       context_lines: 5
       description: "My custom error type"
   ```

2. Use it:
   ```bash
   iree-ci-triage --run 12345 --patterns my_patterns.yaml
   ```

### Custom Root Cause Rules

1. Create `my_rules.yaml`:
   ```yaml
   root_cause_rules:
     - name: "my_root_cause"
       primary_pattern: "my_error"
       secondary_patterns: ["timeout"]
       priority: 70
       actionable: true
       category: "custom"
   ```

2. Use it:
   ```bash
   iree-ci-triage --run 12345 --rules my_rules.yaml
   ```

---

## Troubleshooting

### "GitHub CLI (gh) not found"

```bash
# Install gh CLI
# See: https://cli.github.com
```

### "GitHub CLI is not authenticated"

```bash
gh auth login
# Follow prompts to authenticate
```

### "Pattern file not found"

Make sure you're using absolute paths or paths relative to the current directory:
```bash
iree-ci-triage --run 12345 --patterns /full/path/to/patterns.yaml
```

### "No failed jobs found"

The workflow run may have no failures, or the run ID is incorrect. Verify:
```bash
gh run view 12345 --repo iree-org/iree
```

---

## Contributing

- Add new patterns: See `PATTERN_GUIDE.md`
- Report issues: File an issue with example run ID showing missed error
- Submit improvements: PRs welcome

---

## See Also

- **PATTERN_GUIDE.md** - Detailed pattern authoring guide
- **patterns.yaml** - All error pattern definitions
- **cooccurrence_rules.yaml** - Root cause grouping rules
