# IREE Developer Utilities

Developer-facing utilities for IREE contributors: testing, CI triage, and analysis.

Distinct from `build_tools/` (building IREE), `tests/` (test suites), and `tools/` (user-facing compiler tools like iree-compile).

## Installation

**1. Install dependencies:**
```bash
pip install -e tools/utils
```

**2. Add to PATH (choose one):**

```bash
# Option A: Export PATH
export PATH="$PATH:$PWD/tools/utils/bin"

# Option B: direnv (recommended)
cp .envrc.example .envrc
direnv allow
```

**3. Verify:**
```bash
iree-lit-list --help
```

## Tools

### Lit Test Tools (`iree-lit-*`)

Work with MLIR lit test files that use `// -----` delimiters.

| Tool | Purpose | Example |
|------|---------|---------|
| `iree-lit-list` | List test cases | `iree-lit-list test.mlir` |
| `iree-lit-extract` | Extract individual cases | `iree-lit-extract test.mlir --case 3` |
| `iree-lit-replace` | Replace case content | `echo "..." \| iree-lit-replace test.mlir --case 3` |
| `iree-lit-test` | Run tests in isolation | `iree-lit-test test.mlir --case 2 --verbose` |
| `iree-lit-lint` | Lint for style issues | `iree-lit-lint test.mlir` |

```bash
# Common workflow: debug a failing test
iree-lit-list test.mlir                    # See structure
iree-lit-test test.mlir --case 2 --verbose # Run failing case
iree-lit-extract test.mlir --case 2        # Extract for manual debugging
iree-lit-lint test.mlir --case 2           # Check style

# Run all tests in a file
iree-lit-test test.mlir

# Lint all modified .mlir files
git diff --name-only HEAD | grep '\.mlir$' | xargs -I{} iree-lit-lint {}
```

### CI Triage Tools (`iree-ci-*`)

Analyze GitHub Actions failures.

| Tool | Purpose | Example |
|------|---------|---------|
| `iree-ci-triage` | Analyze CI failures | `iree-ci-triage --pr 12345` |
| `iree-ci-garden` | Manage failure corpus | `iree-ci-garden --help` |

```bash
# Triage all CI failures for a PR
iree-ci-triage --pr 12345

# Triage a specific workflow run
iree-ci-triage --run 9876543210

# Get JSON output for scripting
iree-ci-triage --pr 12345 --json
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `IREE_BUILD_DIR` | Override build directory detection (for `iree-lit-test`) |

## File System Hygiene

These tools never write to the source tree:
- No temporary files in test directories
- All temp files use system temp directories
- Output files only where explicitly specified (`-o`, `--json-output`)

## Documentation

- `--help` on any tool for detailed usage
- `iree-lit-lint --help-style-guide` for MLIR test style guide
- `lit_tools/STYLE_GUIDE.md` for full style documentation

## Claude Code Integration

Claude commands are available for these tools:
- `/iree-ci-triage {pr}` - Triage CI failures
- `/iree-lit-test {file}` - Run and debug a lit test
- `/iree-lit-lint {file}` - Lint a lit test

The `iree-lit-tools` skill provides guidance for MLIR test authoring.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines:
- Code style and authoring requirements
- Testing requirements (356 tests must pass)
- Import strategy and execution model
- Adding new tools

```bash
# Run all tests
python -m unittest discover -s tools/utils -p "test_*.py" -v
# Expected: Ran 356 tests ... OK
```
