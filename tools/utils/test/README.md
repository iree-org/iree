# IREE Lit Tools Test Suite

This directory contains comprehensive tests for the iree-lit-* utilities (iree-lit-test, iree-lit-extract, iree-lit-replace, iree-lit-list).

## Running Tests

Tests work from any location using unittest discovery:

```bash
# From IREE repo root (primary workflow - works everywhere):
python -m unittest discover -s tools/utils -p "test_*.py" -v

# From tools/utils directory (local development):
cd tools/utils
python -m unittest discover -s . -p "test_*.py" -v

# From any location (absolute path):
python -m unittest discover -s /absolute/path/to/iree/tools/utils -p "test_*.py" -v
```

This discovers and runs:
- Unit tests in `lit_tools/tests/` (40 tests)
- Integration tests in `test/lit_tests/` (316 tests)
- **Total: 356 tests**

### Common Issues

**Build detection failures:** Some tests require `iree-opt` and other tools. Set `IREE_BUILD_DIR` or ensure tools are in PATH.

## Test Organization

### `lit_tools/tests/` - Unit Tests
- `test_suggestions.py` - Fuzzy matching for case names
- `test_verification.py` - IR verification with iree-opt (18 tests)
- `test_suggestions_integration.py` - Integration tests for suggestions

### `test/lit_tests/` - Integration Tests
- `test_core_cli.py` - CLI argument parsing, case selection, filtering
- `test_extract.py` - iree-lit-extract functionality
- `test_replace.py` - iree-lit-replace functionality
- `test_replace_integration.py` - End-to-end workflows
- `test_list.py` - iree-lit-list functionality
- `test_lit_test_cli.py` - iree-lit-test CLI integration
- `test_integration_iree_lit_test.py` - Full tool integration
- `test_workflows.py` - Multi-tool workflows
- Additional specialized test files

## Test Coverage

The Python test suite provides comprehensive coverage:

### ✅ Basic Functionality
- Timeout protection
- Extra flags injection
- Verbose/quiet modes
- Multiple output formats (text, JSON)

### ✅ Case Selection
- Single case by number/name
- Comma-separated lists (`--case 1,3,5`)
- Range syntax (`--case 1-3`, `--case 5-10`)
- Mixed syntax (`--case 1,3-5,7`)
- Multiple --case flags
- Line number selection (`--containing`)

### ✅ Filtering
- Regex include (`--filter`)
- Regex exclude (`--filter-out`)
- Combined filtering

### ✅ Modes
- List mode (`--list`)
- Dry-run mode (`--dry-run`)
- Extraction modes (stdout, file output, JSON)

### ✅ Error Handling
- File not found
- Invalid case numbers/names
- Invalid ranges
- Filter no matches
- Missing required tools

### ✅ Edge Cases
- Whitespace in arguments
- Empty arguments
- Zero and negative case numbers
- Duplicate case numbers (auto-deduplicated)
- Very large case ranges

### ✅ Tool Integration
- Extract → edit → replace workflows
- JSON mode for automation
- Multi-file operations

## For LLMs and Automation

**Single source of truth for testing:**

```bash
# From IREE repo root (works everywhere):
python -m unittest discover -s tools/utils -p "test_*.py" -v
```

Expected output on success:
```
Ran 356 tests in X.XXXs

OK
```

**Never declare tests passing without running the full suite (356 tests).**

## Adding New Tests

1. **Unit tests** for core modules → `lit_tools/tests/test_*.py`
2. **Integration tests** for CLI tools → `test/lit_tests/test_*.py`
3. Use fixtures from `test/lit_tests/fixtures/`
4. Follow existing patterns (context managers, mocking with `patch()`)

## Subprocess Best Practices

### CRITICAL: Use `run_python_module()` for All Python Module Invocations

**When writing integration tests**, always use the `run_python_module()` helper from `test.test_helpers`:

```python
from test.test_helpers import run_python_module

# ✅ CORRECT: Use run_python_module for tools/utils modules
result = run_python_module(
    "lit_tools.iree_lit_extract",
    [str(test_file), "--case", "2"],
    capture_output=True,
    text=True,
)

# ✅ CORRECT: External binaries use subprocess.run directly
result = subprocess.run(
    ["gh", "pr", "view", "12345"],
    capture_output=True,
)

# ❌ WRONG: Hardcoded python3 - breaks in venv, pyenv, etc.
result = subprocess.run(
    ["python3", "-m", "lit_tools.iree_lit_extract", ...],
    ...
)
```

### Why run_python_module()?

The helper ensures tests work correctly in ALL environments:

1. **Uses current Python interpreter** (`sys.executable`) instead of hardcoded `"python3"`
   - Works with pyenv, venv, virtualenv, different Python versions
2. **Sets up PYTHONPATH** automatically for tools/utils imports
   - Tests work from repo root AND from tools/utils directory
3. **Preserves environment** - existing PYTHONPATH, env vars are maintained
4. **Future-proof** - new tests automatically use correct patterns

### When to Use What

```python
# Python modules in tools/utils (lit_tools, ci, common):
from test.test_helpers import run_python_module
result = run_python_module("lit_tools.iree_lit_list", [str(file)])

# External binaries (gh, git, iree-opt, iree-compile):
import subprocess
result = subprocess.run(["gh", "pr", "list"], capture_output=True)
```

### Examples

```python
# Extract a test case to JSON
result = run_python_module(
    "lit_tools.iree_lit_extract",
    [str(test_file), "--case", "2", "--json"],
    capture_output=True,
    text=True,
)
extracted = json.loads(result.stdout)

# Replace with stdin input
result = run_python_module(
    "lit_tools.iree_lit_replace",
    [str(test_file), "--case", "1"],
    input=new_content,
    capture_output=True,
    text=True,
)

# Run CI triage tool
result = run_python_module(
    "ci.iree_ci_triage",
    ["--pr", "12345", "--verbose"],
    capture_output=True,
    text=True,
)
```

All `subprocess.run()` kwargs work: `capture_output`, `text`, `input`, `timeout`, `check`, etc.

## Test Fixtures

Located in `test/lit_tests/fixtures/`:
- `simple_test.mlir` - Single test case
- `split_test.mlir` - Three test cases with `// -----` separators
- `ten_cases_test.mlir` - Ten test cases for multi-case selection testing
- Additional specialized fixtures

All fixtures use valid MLIR syntax and can be processed by iree-opt.
