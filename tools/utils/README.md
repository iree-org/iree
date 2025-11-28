# IREE Developer Utilities

Swiss army knife for working on IREE: testing, CI triage, benchmarking, build analysis.

## Pre-commit (Utils)

This subtree has a single, scoped pre-commit hook that runs only on `tools/utils/`:

- Hook id: `utils-validate` (cross-platform, Python)
- Checks:
  - Permissions: only `tools/utils/bin/*` are executable; wrappers must have a python3 shebang
  - Imports: no `from lit ...` (use `from lit_tools ...`), except in the lit integration code/tests
  - JSON purity: never `print()` to stdout from JSON branches (`if args.json:`)
- Run manually:
  - `pre-commit run utils-validate -a`
  - or `python3 tools/utils/scripts/precommit_utils.py`

## What This Is

Developer-facing tools for IREE contributors. Distinct from:
- `build_tools/` - Infrastructure for building IREE itself
- `tests/` - Actual test suites
- `tools/` - User-facing compiler tools (iree-compile, iree-run-module)

These utilities help developers debug, analyze, and fix issues during development.

## Tool Categories

- **`lit/`** - Lit test utilities (extract cases, run tests, fix failures)
- **`ci/`** - CI triage tools (fetch logs, parse failures, correlate with changes)
- **`runtime/`** - Runtime testing tools (gtest, benchmarks, traces)
- **`build/`** - Build analysis tools (cmake validation, build profiling)
- **`common/`** - Shared infrastructure (build detection, process utils)

## Usage

Tools follow pattern: `iree-<category>-<action>`

```bash
# Lit testing
iree-lit-list test.mlir                 # List test cases in file
iree-lit-extract test.mlir --case 3     # Extract test case #3
iree-lit-replace test.mlir --case 3     # Replace test case #3 (from stdin)
iree-lit-test test.mlir --case 2        # Run test case #2

# CI triage
iree-ci-fetch PR 12345               # Fetch CI logs from PR
iree-ci-parse failure.log            # Parse failure patterns

# Use --help for detailed usage
iree-lit-list --help
```

## Installation

Add to PATH:
```bash
export PATH="$PATH:/path/to/iree/tools/utils/bin"
```

Or run directly:
```bash
python3 tools/utils/lit_tools/iree_lit_list.py --help
```

## File System Hygiene

The lit test utilities never write to the source tree:
- **No temporary files** created in test directories
- **No generated files** in source paths
- **All temporary files** use system temp directories (`/tmp` or `tempfile.TemporaryDirectory()`)
- **Output files** (`--json-output`, `-o`) only go where explicitly specified by user

This prevents git pollution and works cleanly in read-only or shared source trees.

## Development Rules

### Code Style
- **Python 3 stdlib only** - No third-party dependencies unless absolutely necessary
- **argparse for CLI** - Consistent with IREE's existing scripts
- **Apache 2.0 + LLVM header** - Include license header in all files
- **Docstrings with examples** - Module docstring shows usage examples
- **Consistent output modes** - All tools should support:
  - `--json` for machine-readable output
  - `--pretty` for human-friendly formatting (opt-in)
  - default mode should be concise and script-friendly

### Authoring Guidelines (Required)

- Always add common output flags with `cli.add_common_output_flags(parser)`.
- Use `console.error/warn/note/success` for all human messages; never `print()` directly.
- Use `console.print_json(...)` for JSON; do not mix JSON and human text on stdout.
- Honor `--quiet`: suppress non-essential text (warnings/notes/success) automatically via `console.*`.
- Return `exit_codes.SUCCESS/ERROR/NOT_FOUND` instead of magic numbers.
- Write files via `fs.safe_write_text(...)` (UTF-8, atomic rename, normalized newlines).
- Shebang policy: only true CLI entrypoints in `tools/utils/bin/` have a `#!/usr/bin/env python3` shebang. All other modules and tests should not.
- For occasional colorization, prefer `formatting.color(code, text, pretty=...)` instead of custom ANSI logic.
- **CRITICAL**: Never disable lint checks (ruff, black) with `# noqa` or similar without explicit approval. Fix the code properly instead.

### Token Efficiency & LLM-Friendliness
- Default output is concise and stable; opt into `--pretty` for humans.
- `--json` prints only JSON to stdout; human notes go to stderr.
- `--quiet` suppresses notes/warnings/success to save tokens.
- Use shared console/formatting helpers to avoid drift and noise.

### JSON Schema Coordination

- In-tree only: producers and consumers live together; update both in the same PR.
- No version fields: keep payloads small; tests catch breakages.
- For large runs, write JSON to a file (`--json-output`) and post-process with `jq`.

### I/O Output Contract (lit tools)

All lit-category tools follow simple, predictable I/O rules:

- Without `-o`:
  - With `--json`: JSON is printed to stdout only.
  - Without `--json`: Text is printed to stdout only.
- With `-o <file>`:
  - With `--json`: JSON is written to `<file>`; stdout remains quiet (diagnostics only).
  - Without `--json`: Text is written to `<file>`; stdout remains quiet (diagnostics only).

For `iree-lit-extract` specifically:
- Text mode subsets multiple selected cases with `// -----` separators.
- RUN header lines are included by default only when writing to a file with `-o`.
  Use `--include-run-lines` to include them when printing to stdout.
- JSON mode returns an array of case objects; each object includes `content`.

Minimal tool skeleton:

```python
import sys
from pathlib import Path

# Add parent directory to sys.path (required for all tools)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from own category (if in lit/, use lit.core)
from lit_tools.core import cli, console, exit_codes

# Import from other categories
from common import fs

parser = argparse.ArgumentParser(...)
cli.add_common_output_flags(parser)
args = parser.parse_args()

try:
    # do work
    if args.json:
        console.print_json({"ok": True}, args=args)
    else:
        console.note("did the thing", args=args)
    return exit_codes.SUCCESS
except FileNotFoundError as e:
    console.error(str(e), args=args)
    return exit_codes.NOT_FOUND
except Exception as e:
    console.error(f"failed: {e}", args=args)
    return exit_codes.ERROR
```

## Pre-commit and Lint

We run pre-commit on these utilities for formatting and consistency:

- Black formats Python files under `tools/utils/`.
- Ruff (optional but enabled) lints and organizes imports for `tools/utils/`.
- A local hook `iree-utils-lint` enforces our conventions (console/cli/exit-codes/fs).

Run manually:

```bash
pre-commit run -a --files $(git ls-files 'tools/utils/**/*.py')
```

Example violations:

```
tools/utils/lit_tools/foo.py:42: use console.error/warn/note instead of print(..., file=sys.stderr)
tools/utils/lit_tools/bar.py:15: use exit_codes.ERROR instead of sys.exit(1)
tools/utils/lit_tools/baz.py: add common flags with cli.add_common_output_flags(parser)
```

If you need to onboard a new repository checkout:

```bash
pre-commit install
```

To intentionally skip the CLI flags check for a non-CLI module, add a one-line pragma:

```python
# lint: disable=cli-flags
```

### Testing Requirements

**CRITICAL: Single test command for ALL tests**

Tests work from any location using unittest discovery:

```bash
# From IREE repo root (primary workflow - works everywhere):
python -m unittest discover -s tools/utils -p "test_*.py" -v

# From tools/utils directory (local development):
cd tools/utils
python -m unittest discover -s . -p "test_*.py" -v
```

This discovers and runs ALL 356 tests:
1. `lit_tools/tests/` - Unit tests for modules (40 tests)
2. `test/lit_tests/` - Integration tests for CLI tools (316 tests)

**Never declare "tests pass" until seeing: `Ran 356 tests ... OK`**

See `test/README.md` for detailed test documentation.

**Key Requirements:**
- **unittest for tests** - Matches IREE's test infrastructure
- **Test coverage >80%** - All shared libraries must have comprehensive tests
- **Fixture-based tests** - Use minimal fixtures in `test/lit_tests/fixtures/` for CI
- **Cross-platform** - All tests must work on Windows, Linux, macOS
- **No shell scripts** - Python only for maximum portability

### Documentation
- **Extensive docstrings** - Should work with `--help` as primary docs
- **README.md per category** - Development guide for contributors
- **Examples in docstrings** - Show realistic usage

### Integration Testing Requirements

#### Two types of testing

**1. Fixture-based tests** (required for CI)
- Place fixtures in `test/<category>_tests/fixtures/`
- Fixtures should cover edge cases: unnamed tests, single case, split cases, large files
- ✅ **This is what you must do for CI to pass**

**2. Real-file validation** (development workflow, not in CI)
- During development, manually validate on ≥5 real IREE test files
- Document validation commands in PR description
- **Do NOT add real-file tests to unittest** (files can change, breaking tests)

#### Why separate fixture tests from real-file validation?

**Fixtures give us**:
- ✅ Reproducible tests in any checkout
- ✅ Control over test content (won't change unexpectedly)
- ✅ CI stability

**Real files give us**:
- ✅ Confidence tool works on actual IREE codebase
- ✅ Detection of real-world edge cases
- ✅ Integration validation

#### Development validation workflow

**Python test suite is comprehensive** - no additional validation scripts needed.

The test suite covers all functionality:
- Basic CLI operations
- Edge cases (whitespace, empty args, invalid input)
- Multi-tool workflows
- JSON output validation
- Error handling

**Before submitting PR:**
```bash
# From repo root (recommended - works in all environments):
python -m unittest discover -s tools/utils -p "test_*.py" -v
```

**Expected output:** `Ran 356 tests ... OK`

See `test/README.md` for test organization and coverage details.

#### Recommended fixtures

Every tool should have fixtures covering:
- Single test case (no `// -----` delimiters)
- Multiple test cases (with `// -----`)
- Named cases (with `CHECK-LABEL`)
- Unnamed cases (no `CHECK-LABEL`)
- Large file (10+ cases)
- Edge cases specific to tool functionality

### Scope Guidelines
- **One tool, one job** - Keep tools focused and composable
- **Command-line first** - Scripts call scripts, JSON/text for data interchange
- **Both human and LLM** - Work from terminal or as Claude Code tool
- **Fail loudly** - Clear error messages, proper exit codes

### JSON Schema Coordination

**All JSON producers and consumers are in-tree** - no external users to maintain backward compatibility for.

**When updating JSON schemas**:
- ✅ Update producers and consumers in the same PR
- ✅ Breaking changes are fine (no backward compatibility needed)
- ✅ Let tests catch incompatibilities
- ❌ No version fields (saves tokens for LLM consumers)
- ❌ No upgrade/migration logic (all tools in same tree)

**Token efficiency**: Keep JSON concise - LLMs consume this output. Use short keys where reasonable without sacrificing clarity.

**If incompatibility detection becomes needed**: Add `"v":1` field only then, not preemptively.

### Import Strategy (REQUIRED)

All tools must follow this pattern to support both direct execution and wrapper usage.

#### For tools in category directories (lit/, ci/, runtime/, build/)

```python
# At top of file, before other imports
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from own category (as absolute path within sys.path)
from lit_tools.core import cli, console, exit_codes  # If you're in lit_tools/core/

# Import from other categories (added to sys.path as top-level packages)
from common import build_detection, fs
from lit_tools.core import test_file
```

#### For test files (test/<category>/test_*.py)

```python
import sys
from pathlib import Path

# Add tools/utils to path (parents[2] from test/category/test_*.py)
sys.path.insert(0, str(Path(__file__).parents[2]))

# Import using full absolute paths
from tools.utils.lit import iree_lit_list
from tools.utils.common import build_detection
from tools.utils.lit.core import test_file
```

#### For bin/ wrappers

```python
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the module directly (no tools.utils prefix)
from lit_tools import iree_lit_list  # Note: import the module, not individual functions

if __name__ == "__main__":
    sys.exit(iree_lit_list.main(iree_lit_list.parse_arguments()))
```

#### Why this pattern?

After `sys.path.insert(0, str(Path(__file__).parent.parent))` adds `tools/utils/` to sys.path, Python treats `lit`, `common`, `ci`, etc. as **top-level packages**. This means:

- ✅ Use `from lit_tools.core import ...` for same-category imports (NOT `from .core import ...`)
- ✅ Use `from common import ...` for cross-category imports (NOT `from tools.utils.common import ...`)

**Why absolute imports work for all contexts**:
- ✅ Direct execution: `python3 tools/utils/lit_tools/iree_lit_list.py test.mlir` works because sys.path manipulation runs first
- ✅ Via wrapper: `tools/utils/bin/iree-lit-list test.mlir` works because wrapper sets up sys.path
- ✅ As module: `python3 -m tools.utils.lit_tools.iree_lit_list test.mlir` works because Python resolves the package structure
- ✅ In tests: `python3 -m unittest tools.utils.test.lit_tests.test_list` works because tests add `tools/utils` to sys.path

**CRITICAL: Do NOT use relative imports after sys.path manipulation**:
- ❌ `from .core import ...` only works when imported as part of a package, NOT for direct execution
- ❌ `from ..common import ...` fails with "attempted relative import with no known parent package"
- ✅ `from lit_tools.core import ...` works in all contexts after sys.path is set up

### Execution Model

#### Primary usage method

Always use wrapper scripts in `bin/` for normal usage:
```bash
tools/utils/bin/iree-lit-list test.mlir
```

Add to PATH for convenience:
```bash
export PATH="$PATH:/path/to/iree/tools/utils/bin"
iree-lit-list test.mlir
```

#### Module structure

**Category modules** (`lit/iree_lit_*.py`, `ci/iree_ci_*.py`, etc.) must include:

```python
if __name__ == "__main__":
    sys.exit(main(parse_arguments()))
```

This allows direct testing during development:
```bash
python3 tools/utils/lit_tools/iree_lit_list.py test.mlir  # Works during development
```

#### File naming conventions

- **Python modules**: Use underscores (`iree_lit_list.py`)
- **CLI wrappers**: Use hyphens (`iree-lit-list`)

Example:
- Module: `tools/utils/lit_tools/iree_lit_extract.py`
- Wrapper: `tools/utils/bin/iree-lit-extract`
- Command: `iree-lit-extract test.mlir --case 2`

#### Executable permissions

**DO NOT** make category modules executable (`chmod +x`) - only wrappers in `bin/` should be executable.

**Shebang policy**: Only `bin/` wrappers get shebangs. Category modules should NOT have shebangs.

Correct:
- `bin/iree-lit-list` - ✅ Has shebang, executable
- `lit_tools/iree_lit_list.py` - ✅ No shebang, not executable

Incorrect:
- `lit/iree_lit_list.py` - ❌ Has shebang (violates policy)

### Test Organization (REQUIRED)

#### Directory structure

Tests mirror category structure flatly in `test/<category>/`:

```
tools/utils/test/<category>/test_<module>.py
```

#### Naming conventions

| Module Location | Test Location | Notes |
|----------------|---------------|-------|
| `lit/iree_lit_list.py` | `test/lit_tests/test_list.py` | Drop `iree_lit_` prefix |
| `lit/iree_lit_extract.py` | `test/lit_tests/test_extract.py` | Drop `iree_lit_` prefix |
| `lit/core/test_file.py` | `test/lit_tests/test_core_test_file.py` | Keep `core_` prefix |
| `lit/core/console.py` | `test/lit_tests/test_core_console.py` | Keep `core_` prefix |
| `common/build_detection.py` | `test/common_tests/test_build_detection.py` | Mirrors structure |
| `ci/iree_ci_fetch.py` | `test/ci/test_fetch.py` | Drop `iree_ci_` prefix |

#### Test file naming pattern

```
test_<module_basename>.py
```

Where `<module_basename>` is:
- The module filename without `iree_<category>_` prefix
- With `core_` prefix preserved for core modules

#### Fixtures

Place test fixtures in `test/<category>_tests/fixtures/`:

```
test/lit_tests/fixtures/
  ├── split_test.mlir          # Multiple cases with // -----
  ├── single_case_test.mlir    # Single case, no delimiters
  ├── unnamed_cases_test.mlir  # Cases without CHECK-LABEL
  └── README.md                # Documents what each fixture tests
```

Fixtures should be named `<scenario>_test.mlir` to indicate what they test.

#### Running tests

```bash
# All tests
python3 -m unittest discover -s tools/utils/test -v

# Category-specific
python3 -m unittest discover -s tools/utils/test/lit_tests -v

# Single test file
python3 -m unittest tools.utils.test.lit_tests.test_list -v

# Single test class
python3 -m unittest tools.utils.test.lit_tests.test_list.TestJSONOutput -v
```

### Build Detection Usage Guidelines

Tools that need to invoke IREE binaries (iree-opt, iree-compile, FileCheck) should use the `build_detection` module to locate build artifacts.

#### When to use `build_detection`

**Use build_detection when**:
- ✅ Tool invokes IREE binaries (`iree-opt`, `iree-compile`, `FileCheck`, `iree-run-module`)
- ✅ Tool needs to know build type (Debug vs Release, assertions enabled)
- ✅ Tool needs to locate LLVM tools (`FileCheck`, `llvm-mc`)

**Skip build_detection when**:
- ❌ Tool only parses/lists/extracts text (no binary execution)
- ❌ Tool is a pure utility (formatting, file manipulation)

**Examples**:
- `iree-lit-list` - ❌ No build detection (only parses text)
- `iree-lit-extract` - ⚠️ Optional build detection (only for `--validate` flag)
- `iree-lit-test` - ✅ Required build detection (must run iree-opt)

#### Error handling patterns

**Pattern 1: Optional tool usage** (truly optional features)

```python
from common import build_detection, fs
from lit_tools.core import console, exit_codes

try:
    iree_opt = build_detection.find_tool("iree-opt")
    # Use iree_opt for validation...
except FileNotFoundError:
    console.warn(
        "Skipping validation (iree-opt not found). "
        "Set IREE_BUILD_DIR to enable validation.",
        args=args
    )
    # Continue without validation
```

Use when tool is **nice-to-have** and not required for the requested operation.

**Pattern 2: Required tool** (must run binary to function)

```python
from common import build_detection, fs
from lit_tools.core import console, exit_codes

try:
    iree_opt = build_detection.find_tool("iree-opt")
except FileNotFoundError as e:
    console.error(
        f"Cannot find iree-opt in build directory.\n{e}",
        args=args
    )
    return exit_codes.NOT_FOUND
```

Use when tool is **required** for operation. `build_detection.find_tool()` provides helpful error messages suggesting how to build IREE.

Note: If a user explicitly requests an operation that requires a tool (e.g., `iree-lit-extract --validate` requires `iree-opt`), treat missing tools as errors.

### Pre-commit Lint Examples (terse)

These examples show what you’ll see if conventions are violated:

```
tools/utils/lit_tools/foo.py:42:8: use console.error/warn/note instead of print(..., file=sys.stderr)
tools/utils/lit_tools/bar.py: add common flags with cli.add_common_output_flags(parser)
tools/utils/lit_tools/core/baz.py: missing standard IREE license header
tools/utils/ci/qux.py:1:1: E402 Module level import not at top of file
```

To skip the CLI flag check in a non-CLI module:

```python
# lint: disable=cli-flags
```

#### Build directory detection

**Search order** (automatic):
1. `IREE_BUILD_DIR` environment variable (override)
2. `./build/` (in-tree build, standard CMake default)
3. `../<worktree>-build/` (worktree pattern: iree-loom → iree-loom-build)
4. `../iree-build/` (main repo build)

**Environment variable override**: `IREE_BUILD_DIR`

```bash
# Override build directory detection
export IREE_BUILD_DIR=/custom/build/path
iree-lit-test test.mlir --case 2
```

**No CLI flag needed**: Keep CLI clean by using environment variable only.

#### Helpful error messages

When build directories or tools are not found, `build_detection` provides actionable error messages:

```
Cannot find build directory. Tried:
  1. /home/user/iree/build (in-tree build)
  2. /home/user/iree-loom-build (worktree build)
  3. /home/user/iree-build (main repo build)

Build IREE first:
  cmake -B build -S . && cmake --build build -j$(nproc)

Or set IREE_BUILD_DIR to specify a custom build location.
```

```
Cannot find tool 'FileCheck' in build directory /home/user/iree/build.

Tried:
  - /home/user/iree/build/tools/FileCheck
  - /home/user/iree/build/bin/FileCheck
  - /home/user/iree/build/llvm-project/bin/FileCheck

Tool may not be built yet. Build IREE with:
  cmake --build /home/user/iree/build --target FileCheck

Or build all tools:
  cmake --build /home/user/iree/build -j$(nproc)
```

#### Documenting in --help

If your tool uses build detection, add to help epilog:

```python
parser = argparse.ArgumentParser(
    description="Tool description",
    epilog="""
Environment variables:
  IREE_BUILD_DIR    Override build directory detection
                    (default: auto-detect from worktree layout)
""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
```

### Adding New Tools
1. Add implementation to appropriate category directory
2. Add comprehensive docstring with examples
3. Write tests (unit + integration). Place them under `tools/utils/test/<category>/`.
4. Create a thin Python wrapper in `bin/` (not a symlink)
5. Update category README.md
6. Run full test suite

## Contributing

See individual category READMEs for development details:
- `lit/README.md` - Lit test tools
- `ci/README.md` - CI triage tools
- `runtime/README.md` - Runtime testing tools
- `build/README.md` - Build analysis tools
