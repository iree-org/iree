# Lit Test Utilities - Development Guide

Developer documentation for contributors working on lit test tools.

## Architecture

**Core libraries** (`lit/core/`):
- `test_file.py` - Parse test files (split by `// -----`, extract cases, RUN lines)

**Tools** (`lit/`):
- `iree_lit_list.py` - List test cases
- `iree_lit_extract.py` - Extract individual cases
- `iree_lit_replace.py` - Replace test cases with new content
- `iree_lit_test.py` - Run tests in isolation
- `iree_lit_fix.py` - Interactive test fixing (PHASE 4)

All tools should prefer consistent flags and outputs. In particular, `--json`
is reserved for machine-readable output where applicable.

## Dependencies

```
lit tools
  ↓
lit/core/* (test_file)
  ↓
common/* (build_detection)
  ↓
Python 3 stdlib only
```

## Development Workflow

### Adding a New Tool

1. **Create implementation** in `lit/iree_<tool_name>.py` (use underscores for Python import)
2. **Add docstring** with extensive examples (see template below)
3. **Write tests** in `tests/test_<tool_name>.py`
4. **Create wrapper** in `bin/iree-<tool-name>` (use hyphens for CLI)
5. **Validate**: Run on real IREE test files

### Running Tests

```bash
# Lit-only tests
python3 -m unittest discover -s tools/utils/test/lit_tests -v

# All utils tests
python3 -m unittest discover -s tools/utils/test -v
```

### Integration Testing

**Development validation** (not in CI):

Use the validation script to test on real IREE files:
```bash
# Validate lit tools on real IREE test files
tools/utils/scripts/validate_lit_tools.sh
```

See `tools/utils/README.md` "Integration Testing Requirements" for details on:
- Fixture-based tests (for CI)
- Real-file validation (for development)
- Why not to add real-file tests to unittest

## Tool Implementation Template

```python
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""One-line description of what this tool does.

Detailed explanation of functionality, use cases, and behavior.

Usage:
  # Basic usage
  iree-<category>-<action> input.mlir

  # With options
  iree-<category>-<action> input.mlir --option value

Examples:
  # Example 1: Common case
  $ iree-<category>-<action> test.mlir
  [expected output]

  # Example 2: With flags
  $ iree-<category>-<action> test.mlir --flag
  [expected output]

Exit codes:
  0 - Success
  1 - Error (invalid input, execution failure, etc.)
  2 - Not found (file doesn't exist, case not found, etc.)

See also:
  iree-related-tool - Related functionality
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from other categories (added to sys.path as top-level packages)
from common import fs

# Import from own category (as absolute path within sys.path)
from lit_tools.core import cli, console, exit_codes, test_file


def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="One-line description",
        epilog="See module docstring for examples and details",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", help="Input test file")
    parser.add_argument("--option", help="Option description")
    cli.add_common_output_flags(parser)
    return parser.parse_args()


def main(args):
    """Main entry point."""
    # Validate input
    file_path = Path(args.file)
    if not file_path.exists():
        console.error(f"File not found: {file_path}", args=args)
        return exit_codes.NOT_FOUND

    # Implementation
    try:
        # Do work
        result = do_work(args)
        if args.json:
            console.print_json({"result": result}, args=args)
        else:
            print(result)
        return exit_codes.SUCCESS
    except Exception as e:
        console.error(f"Error: {e}", args=args)
        return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main(parse_arguments()))
```

### Binary Wrapper Pattern (tools/utils/bin/)

For cross-platform compatibility, create thin Python wrappers instead of symlinks:

```python
#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Thin wrapper for iree-<tool-name> tool."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and run the actual tool
from lit_tools import iree_tool_name  # Note: underscores in Python module name

if __name__ == "__main__":
    sys.exit(iree_tool_name.main(iree_tool_name.parse_arguments()))
```

**Naming Convention**:
- Python module: `iree_<tool>_<name>.py` (underscores for import)
- CLI wrapper: `iree-<tool>-<name>` (hyphens for command line)

**Execution Methods**:
- **Primary**: Use wrapper in `bin/` (e.g., `iree-lit-list test.mlir`)
- **Development**: Direct module execution (e.g., `python3 tools/utils/lit_tools/iree_lit_list.py test.mlir`)
- **Testing**: Module import (e.g., `python3 -m unittest tools.utils.test.lit_tests.test_list`)

**Shebang Policy**:
- ✅ Wrappers in `bin/` MUST have `#!/usr/bin/env python3`
- ❌ Category modules (`lit/*.py`) MUST NOT have shebangs

## Code Review Checklist

Before submitting:
- [ ] Docstring includes usage examples and exit codes
- [ ] Unit tests added with >80% coverage (using fixtures)
- [ ] Manual validation on real IREE files passes (run validation script)
- [ ] `--help` output is clear and comprehensive
- [ ] Error messages are descriptive (what failed, why, how to fix)
- [ ] Exit codes follow conventions (0=success, 1=error, 2=not found)
- [ ] Follows IREE style (stdlib only, argparse, Apache 2.0 header)
- [ ] Binary wrapper created in `bin/` (not symlink)

## Common Patterns

### Parsing Test Files

```python
# Import from own category (if in lit/ tool, after sys.path manipulation)
from lit_tools.core import test_file

# Parse all cases
cases = test_file.parse_test_file(Path('test.mlir'))

# Extract specific case
case = test_file.extract_case_by_number(Path('test.mlir'), 2)
case = test_file.extract_case_by_name(Path('test.mlir'), 'function_name')

# Get RUN lines
run_lines = test_file.extract_run_lines(Path('test.mlir'))
```

### Build Detection

Lit tools that need to invoke IREE binaries should use the `build_detection` module.

#### When to use build detection

**Lit-specific examples**:
- `iree-lit-list` - ❌ No build detection needed (only parses text)
- `iree-lit-extract` - ⚠️ Optional (only for `--validate` flag)
- `iree-lit-test` - ✅ Required (must run iree-opt and FileCheck)
- `iree-lit-fix` - ✅ Required (must run tests and capture output)

#### Error handling patterns for lit tools

**Optional validation** (iree-lit-extract --validate):

```python
from common import build_detection
from lit_tools.core import console

# Validation is optional, gracefully degrade if tool not found
if args.validate:
    try:
        iree_opt = build_detection.find_tool("iree-opt")
        # Validate IR with iree_opt...
        if not validate_ir(content, iree_opt):
            console.error("IR validation failed", args=args)
            return exit_codes.ERROR
    except FileNotFoundError:
        console.warn("Skipping validation (iree-opt not found)", args=args)
        # Continue without validation
```

**Required tool** (iree-lit-test):

```python
from common import build_detection
from lit_tools.core import console, exit_codes

# Tool is required for core functionality
try:
    iree_opt = build_detection.find_tool("iree-opt")
    filecheck = build_detection.find_tool("FileCheck")
except FileNotFoundError as e:
    console.error(
        f"Cannot run lit tests without IREE build.\n{e}",
        args=args
    )
    return exit_codes.NOT_FOUND

# Run test with iree_opt...
```

**Build type detection** (for debug vs release behavior):

```python
from common import build_detection
from lit_tools.core import console

build_dir = build_detection.detect_build_dir()
build_type = build_detection.get_build_type(build_dir)
is_debug = build_detection.is_debug_build(build_dir)

if is_debug:
    console.note(f"Using {build_type} build with assertions", args=args)
else:
    console.note(f"Using {build_type} build", args=args)
```

#### Build directory detection

**Automatic detection order**:
1. `IREE_BUILD_DIR` environment variable (override)
2. `./build/` (in-tree build, standard CMake default)
3. `../<worktree>-build/` (worktree pattern)
4. `../iree-build/` (main repo build)

**Helpful error messages**: When tools are not found, build_detection provides actionable guidance:
```
Cannot find tool 'iree-opt' in build directory /home/user/iree/build.

Tried:
  - /home/user/iree/build/tools/iree-opt
  - /home/user/iree/build/bin/iree-opt
  - /home/user/iree/build/llvm-project/bin/iree-opt

Tool may not be built yet. Build IREE with:
  cmake --build /home/user/iree/build --target iree-opt

Or build all tools:
  cmake --build /home/user/iree/build -j$(nproc)
```

**Environment variable override**:
```bash
export IREE_BUILD_DIR=/custom/build/path
iree-lit-test test.mlir --case 2
```

See `tools/utils/README.md` for complete build detection documentation.

## Edge Cases to Handle

### Multiple CHECK-LABELs in One Case

Some test cases have multiple CHECK-LABELs. Our parser extracts the **first** one as the case name:

```mlir
// -----

// CHECK-LABEL: @main_function
util.func @main_function() {
  // CHECK-LABEL: @nested_region
  util.func @nested_region() {
    // ...
  }
}
```

Case name will be `main_function` (first CHECK-LABEL).

### No Functions

Some test cases have no functions at all (testing globals, executables, etc.):

```mlir
// -----

stream.executable @executable {
  stream.executable.export @dispatch
  // ...
}
```

These cases show as `(unnamed)` since there's no CHECK-LABEL.

### Mixed Named/Unnamed Cases

A file can have both:
- Cases 1-2: Named functions with CHECK-LABELs
- Cases 3-5: Unnamed (no CHECK-LABEL or no functions)

This is normal and supported.

## Testing Philosophy

### Test Coverage Requirements

- **Unit tests**: Core logic in `lit/core/` must have >80% coverage
- **Fixture-based tests**: Use minimal fixtures in `test/lit_tests/fixtures/` for CI
- **Development validation**: Manually validate on ≥5 real IREE test files using validation script
- **Edge cases**: Test unnamed cases, single case files, large files (10+ cases)

**IMPORTANT**: Do NOT add real-file tests to unittest (files can change, breaking CI). Use fixtures for CI tests and validation script for development.

### Example Fixture Test

```python
def test_parse_split_file(self):
    """Test parsing file with multiple test cases (using fixture)."""
    fixture = self.fixtures_dir / "split_test.mlir"
    cases = test_file.parse_test_file(fixture)
    # Fixture has exactly 3 test cases
    self.assertEqual(len(cases), 3)
    # All should be named
    self.assertTrue(all(case.name for case in cases))
```

Fixtures are stable, reproducible, and won't break when real IREE files change.

## Future Tools (Roadmap)

### Phase 2 (Next)
- `iree-lit-extract` - Extract test cases
- `iree-lit-test` - Run tests with debug build detection

### Phase 3
- `iree-lit-diff` - Compare actual vs expected
- `iree-lit-capture` - Batch capture test outputs

### Phase 4
- `iree-lit-fix` - Interactive test fixing TUI
- `iree-lit-generate-checks` - Auto-generate CHECK patterns

Each phase builds on the previous foundation.
## JSON Output Conventions

- `iree-lit-list --json` prints full metadata for all cases with fields:
  - `file`, `count`, and an array `cases[]` containing: `number`, `name`,
    `start_line`, `end_line`, `line_count`, `check_count`.
- `iree-lit-extract --list --json` matches `iree-lit-list --json` semantics.
- `iree-lit-extract <selector> --json` prints an array of case objects to
  stdout. With `-o <file>`, the JSON array is written to `<file>` instead of
  stdout.
## Output Modes (Standardized)

All lit tools should support consistent output modes:
- `--json` emits machine-readable JSON. Prefer this for automation/LLMs.
- Default text output should be concise and stable.
- `--pretty` enables human-friendly formatting (and optional color when TTY).

JSON schema for listings:
```json
{
  "file": "path/to/test.mlir",
  "count": 3,
  "cases": [
    {"number": 1, "name": "foo", "start_line": 1, "end_line": 12, "line_count": 12, "check_count": 3}
  ]
}
```

JSON schema for extraction (array of cases):
```json
[
  {
    "number": 2,
    "name": "second_case",
    "start_line": 14,
    "end_line": 24,
    "line_count": 11,
    "check_count": 4,
    "content": "// CHECK-LABEL: @second_case\nutil.func @second_case() {\n  ...\n}"
  }
]
```

### Text Mode Subsetting

- When writing to a file with `-o`, RUN lines from the file header are
  included by default at the top, followed by the selected cases separated by
  `// -----`. This makes the output a proper subset file of the original.
- When printing to stdout, RUN lines are not added by default; pass
  `--include-run-lines` to include them in stdout output.

### JSON Schema Coordination

**In-tree only**: All producers/consumers are in the same tree. Update both in the same PR.

**No versioning**: No version fields needed - saves tokens for LLM consumers. Breaking changes are fine; tests catch incompatibilities.

See `tools/utils/README.md` "JSON Schema Coordination" for full guidance.

## Integration Flows (Cross-Tool)

These flows are exercised by tests under `tools/utils/test/lit_tests/`:

- List → Extract:
  - `iree-lit-list file.mlir --names | xargs -n1 -I{} iree-lit-extract file.mlir --name {}`
  - `iree-lit-list file.mlir --json | jq` to select cases, then `iree-lit-extract --case ... --json -o subset.json`.

- List → Test:
  - `iree-lit-list file.mlir --count` equals `iree-lit-test file.mlir --dry-run --json | jq .total_cases`.
  - Drive `iree-lit-test --case ... --dry-run --json` using selections from `iree-lit-list --json`.

- Extract → Test:
  - `iree-lit-extract file.mlir --case 1,3 -o subset.mlir` then `iree-lit-test subset.mlir --dry-run --json` (renumbering begins at 1).

Planned flows
- Replace (future): `iree-lit-replace` to edit cases or CHECK lines in subsets, then run `iree-lit-test`.
  Tests are stubbed and skipped until the tool lands.

## iree-lit-test: Run LIT Tests in Isolation

**For user documentation**, run `iree-lit-test --help`. This section contains contributor documentation, troubleshooting, and implementation details.

### Why iree-lit-test?

**The Problem**: IREE's MLIR test files use `// -----` to pack multiple test cases into single files. When a test fails, developers must:
1. Manually isolate the failing case
2. Copy it to a temporary file
3. Re-run just that case
4. Debug the failure
5. Copy the fix back
6. Re-run the full file to ensure nothing broke

This workflow is:
- **Slow**: Manual copy-paste is error-prone and breaks flow
- **Fragile**: Line numbers in error messages don't match after extraction
- **Limited**: Can't easily inject debug flags without editing test files
- **Painful**: Debugging 1 of 20 cases means running all 20 every time

**The Solution**: `iree-lit-test` runs individual cases directly:

```bash
# Before: Manual workflow
$ lit test.mlir                       # Fails on case 7
$ edit test.mlir                      # Extract case 7 manually
$ cp case7.mlir /tmp/debug.mlir       # Create temp file
$ iree-opt /tmp/debug.mlir            # Debug (line numbers wrong!)
$ edit /tmp/debug.mlir                # Add --debug flags manually
$ iree-opt --debug /tmp/debug.mlir    # Re-run with flags
$ edit test.mlir                      # Copy fix back
$ lit test.mlir                       # Verify (runs all 20 cases)

# After: With iree-lit-test
$ iree-lit-test test.mlir --case 7 --extra-flags="--debug" --verbose
# Runs case 7 only, with debug flags, shows full output
```

**Key Benefits**:
- **Instant iteration**: Run single cases in <1s vs entire suite in 10s+
- **Accurate line numbers**: Errors show correct line numbers from original file
- **No file editing**: Inject debug flags without touching test files
- **Better diagnostics**: Automatically detects crashes, timeouts, IR errors, assertions
- **Parallel execution**: Run multiple cases concurrently with `--workers`
- **Script-friendly**: JSON output for automation, `--quiet` for CI

**When to use**:
- ✅ Debugging a specific failing test case
- ✅ Iterating on a compiler pass with --debug flags
- ✅ Reproducing a flaky test quickly
- ✅ Running subset of cases during development
- ✅ Getting better error diagnostics than raw lit

**When NOT to use**:
- ❌ Running full test suite (use `lit` or `ctest`)
- ❌ Modifying test files (use editor)
- ❌ Generating new CHECK patterns (use update scripts)

### Implementation Details

**Programmatic LIT integration**: We import and drive LLVM's lit APIs in-process:

1. Build a temporary test shard under `/tmp/iree_lit_test_$PID/`.
2. Preserve original line numbers by prepending `(start_line - 1)` blanks.
3. Re-inject RUN lines:
   - Header RUN lines from the file header.
   - Case-local RUN lines that appear inside the case body (rare, but supported).
4. Resolve lit configuration via `config_map` to the real suite `lit.cfg.py` in the source tree (no writes to the source tree).
5. Execute tests with `lit.discovery` and `lit.run.Run`, capture results.

**Benefits**:
- Leverages battle-tested LIT infrastructure (RUN line parsing, substitutions, pipelines)
- Automatically handles `--split-input-file`, `--verify-diagnostics`, multi-line RUN, etc.
- Compatible with all `lit.cfg.py` configurations
- Changes to LIT or lit.cfg.py "just work"

**Line number preservation**: Prepends `(start_line - 1)` blank lines and re-injects RUN lines so FileCheck reports correct line numbers:

```
Original: Case 2 starts at line 15
Extracted: "\n" * 14 + case_content
FileCheck error at stdin:20 = original file line 20 ✓
```

### Advanced Workflows

**Use with git bisect**:
```bash
# Create bisect script.
$ cat > test_bisect.sh << 'BISECT_EOF'
#!/bin/bash
cmake --build ../iree-build -j90 --target iree-opt || exit 125
iree-lit-test compiler/src/iree/compiler/Dialect/Stream/Transforms/test/elide_async_copies.mlir \
    --case 3 --quiet
BISECT_EOF

$ chmod +x test_bisect.sh
$ git bisect start HEAD good_commit
$ git bisect run ./test_bisect.sh
```

**CI/scripting usage**:
```bash
# Quiet mode for scripts (only errors shown).
$ iree-lit-test test.mlir --quiet
$ echo $?  # 0 = pass, 1 = fail, 2 = not found

# JSON mode for automation.
$ iree-lit-test test.mlir --json > results.json
$ if jq -e '.failed == 0' results.json > /dev/null; then
    echo "All tests passed"
  fi
```

### Troubleshooting Guide

**Problem: "Cannot find build directory"**

```
Error: Cannot find build directory. Tried:
  1. /home/user/iree/build (in-tree build)
  2. /home/user/iree-loom-build (worktree build)
  3. /home/user/iree-build (main repo build)
```

**Solution**:
1. Build IREE first: `cmake -B build -S . && cmake --build build -j$(nproc)`
2. Or set `IREE_BUILD_DIR` environment variable:
   ```bash
   export IREE_BUILD_DIR=/path/to/your/build
   iree-lit-test test.mlir
   ```
3. Or pass `--build-dir`:
   ```bash
   iree-lit-test test.mlir --build-dir /path/to/your/build
   ```

**Problem: "Cannot find tool 'iree-opt'"**

```
Error: Cannot find tool 'iree-opt' in build directory.
Tool may not be built yet.
```

**Solution**:
1. Build the specific tool: `cmake --build ../iree-build --target iree-opt`
2. Or build all tools: `cmake --build ../iree-build -j$(nproc)`
3. Verify tool exists: `ls ../iree-build/tools/iree-opt`

**Problem: Test hangs forever**

**Symptoms**: Test runs but never completes.

**Solution**:
1. Kill with Ctrl+C
2. Re-run with shorter timeout: `iree-lit-test test.mlir --case 5 --timeout 10`
3. If still hangs, disable timeout and debug: `iree-lit-test test.mlir --case 5 --timeout 0 --verbose`
4. Check for infinite loops in compiler pass with `--extra-flags="--debug"`

**Problem: Wrong line numbers in error messages**

**Expected**: Error at line 42
**Actual**: Error at line 1

**Cause**: This shouldn't happen with iree-lit-test (it preserves line numbers).

**Solution**:
1. Verify you're using `iree-lit-test`, not raw `iree-opt` on extracted file
2. If using `--keep-temps`, check that temp file has blank line padding
3. Report as bug if line numbers are still wrong

**Problem: Case not found by name**

```
Error: No case found with name 'my_function'
```

**Solution**:
1. List cases: `iree-lit-list test.mlir`
2. Verify case name spelling (case-sensitive)
3. Use `@` prefix for name: `--name my_function` not `--name @my_function`
4. Or use case number: `--case 5`

**Problem: All cases fail with "lit not found"**

```
Error: Cannot import LLVM lit module.
Please ensure IREE was built with LLVM (not installed via package manager).
```

**Solution**:
1. IREE must be built from source with LLVM submodule
2. Cannot use system-installed LLVM packages
3. Rebuild IREE with bundled LLVM: `cmake -B build -S . -DIREE_BUILD_BUNDLED_LLVM=ON`

**Problem: FileCheck errors are truncated**

**Symptoms**: Error message cuts off mid-line.

**Solution**:
1. Use `--verbose` to see full lit output: `iree-lit-test test.mlir --case 5 --verbose`
2. Or use `--keep-temps` and inspect temp file: `iree-lit-test test.mlir --case 5 --keep-temps`
3. Full output includes complete FileCheck diagnostics

**Problem: Can't reproduce lit failure**

**Symptoms**: `lit test.mlir` fails, but `iree-lit-test test.mlir` passes.

**Possible causes**:
1. Test relies on running all cases in sequence (rare, usually a bug)
2. Race condition or test pollution between cases
3. Different lit configuration being used

**Solution**:
1. Run all cases: `iree-lit-test test.mlir` (no `--case`)
2. Compare lit vs iree-lit-test output carefully
3. Report as bug if reproducible

**Problem: Parallel execution is flaky**

**Symptoms**: `--workers 4` sometimes fails, `--workers 1` always passes.

**Cause**: Tests may have shared state or race conditions.

**Solution**:
1. Run serially: `iree-lit-test test.mlir --workers 1`
2. Check if test uses global state or temp files without unique names
3. Fix test to be parallel-safe (use unique temp file names per case)

**Problem: JSON output is mixed with text**

**Symptoms**: `--json` output contains non-JSON text.

**Cause**: Progress messages are mixed with JSON output.

**Solution**:
1. Use `--quiet` with `--json`: `iree-lit-test test.mlir --json --quiet`
2. Or write to file: `iree-lit-test test.mlir --json --json-output results.json`

**Problem: Test passes locally but fails in CI**

**Possible causes**:
1. Different build configuration (Release vs Debug, different flags)
2. Different LLVM version or patches
3. Missing tools in CI build
4. Timeout too short for slower CI machines

**Solution**:
1. Match CI build configuration locally
2. Check CI logs for specific error (use `--verbose` in CI)
3. Increase timeout in CI: `--timeout 120`
4. Ensure all required tools are built in CI

**Problem: Segmentation fault or crash**

**Symptoms**: Test crashes with SIGSEGV or similar.

**Solution**:
1. Run with verbose output: `iree-lit-test test.mlir --case 5 --verbose`
2. Build in debug mode for better stack traces
3. Use gdb: `gdb --args iree-opt <command from --verbose output>`
4. Report crash with stack trace

### Advanced: JSON for LLMs and Large Outputs

`--json` returns a compact per-case summary by default. Enable `--full-json` to include full per-case output (can be very large). Prefer writing JSON to a file with `--json-output` for heavy runs and process with `jq`:

```bash
# Full output for debugging (large):
iree-lit-test test.mlir --json --full-json --json-output /tmp/out.json

# Extract failing cases and their commands:
jq '.results[] | select(.passed==false) | {n:.case_number, name:.case_name, cmd:.run_commands[-1]}' /tmp/out.json

# Count failures quickly:
jq '[.results[] | select(.passed==false)] | length' /tmp/out.json
```

Token efficiency guidance:
- Default JSON omits massive `output` fields; opt-in with `--full-json`.
- For large suites or verbose runs, always use `--json-output` and post-process.
- Expect that `--full-json` for big tests (multi-stage pipelines) can exceed 10k+ lines per case.

### See Also

- `iree-lit-list` - List test cases in file
- `iree-lit-extract` - Extract test case content
- `iree-lit-replace` - Replace test case content
- Validation script: `tools/utils/scripts/validate_lit_test.sh`

## iree-lit-replace: Replace Test Case Content

**For user documentation**, run `iree-lit-replace --help`. This section contains contributor documentation, implementation details, and troubleshooting.

### Why iree-lit-replace?

**The Problem**: Updating test cases in multi-case MLIR files requires careful manual editing:
1. Locate the specific case to update
2. Manually replace content without breaking delimiters
3. Ensure RUN lines are preserved correctly
4. Avoid corrupting adjacent cases
5. Verify case numbering remains consistent

This workflow is:
- **Error-prone**: Easy to corrupt delimiters or adjacent cases
- **Tedious**: Manual editing for batch updates across multiple cases
- **Fragile**: Line numbers shift, making error messages confusing
- **Limited**: No programmatic way to update test outputs

**The Solution**: `iree-lit-replace` atomically replaces test case content:

```bash
# Before: Manual workflow
$ edit test.mlir                      # Manually find and replace case 3
$ lit test.mlir                       # Hope you didn't break delimiters

# After: With iree-lit-replace
$ iree-lit-extract test.mlir --case 3 | edit_pipeline | iree-lit-replace test.mlir --case 3
# Or from a file
$ iree-lit-replace test.mlir --case 3 < new_content.mlir
```

**Key Benefits**:
- **Safe**: Atomic writes with backup files, validates case boundaries
- **Precise**: Replace by case number or name, not line numbers
- **Batch-friendly**: Replace multiple cases in one operation (JSON mode)
- **Validated**: Optional IR validation with `--validate`
- **Preview**: Dry-run mode shows diffs before writing

### Usage Patterns

**Basic replacement** (text mode):
```bash
# Replace case 3 with content from stdin
iree-lit-replace test.mlir --case 3 < new_content.mlir

# Replace by name
iree-lit-replace test.mlir --name my_function < new_content.mlir

# Replace with validation
iree-lit-replace test.mlir --case 2 --validate < new_content.mlir
```

### Validation Features

`iree-lit-replace` performs strict validation to prevent common errors during test case replacement:

**Name/number consistency checking**:
- When JSON includes both `name` and `number` fields, they must point to the same test case
- Catches copy-paste errors where metadata becomes inconsistent
- Prevents targeting the wrong case due to mismatched identifiers

```bash
# Error example:
# JSON: {"number": 2, "name": "foo", ...}
# But case 2 is actually named "bar"
Error: Name/number mismatch: 'number' 2 and 'name' 'foo' don't match same case
  (number points to case 2, name points to case 5)
Fix: Remove one field or ensure they match.
```

**Duplicate case name handling**:
- If multiple cases share the same CHECK-LABEL name, you must disambiguate
- Text mode: Use `--case NUMBER` instead of `--name`
- JSON mode: Use `"number"` field instead of `"name"`
- Prevents accidentally replacing the wrong case

```bash
# Error example when file has two cases both named "my_function":
Error: Multiple cases named 'my_function' found at numbers [2, 5].
Use --case NUMBER to specify which one.

# Or in JSON mode:
Error: Multiple cases named 'my_function' found at numbers [2, 5].
Use 'number' field instead of 'name' to disambiguate.
```

**Duplicate replacement entry detection**:
- JSON mode rejects multiple replacement entries targeting the same case
- Prevents unintentional overwrites from duplicate entries
- Each case can only be replaced once per operation

```bash
# Error example:
Error: Duplicate replacement for number 2: entries 3 and 7.
Fix: Remove duplicate entries, keep only one.
```

**CLI file override warnings**:
- When `--test-file` CLI argument is used with JSON mode, warns if JSON entries have different `"file"` values
- Helps catch unintentional file overrides
- Suppressed with `--quiet` flag

```bash
# Warning example:
Warning: CLI argument 'target.mlir' overriding JSON 'file' field for 3 replacement(s): cases 1, 2, 5
```

All validation errors include actionable "Fix:" suggestions to help resolve issues quickly.

**Dry-run** (preview changes):
```bash
# See what would change (with diff)
iree-lit-replace test.mlir --case 3 --dry-run < new_content.mlir

# Dry-run with pretty diff colors
iree-lit-replace test.mlir --case 3 --dry-run --pretty < new_content.mlir

# JSON output for dry-run
iree-lit-replace test.mlir --case 3 --dry-run --json < new_content.mlir
```

**Batch replacement** (JSON mode):
```bash
# Replace multiple cases from JSON array
cat replacements.json | iree-lit-replace --mode json

# JSON format:
[
  {
    "file": "test/foo.mlir",
    "number": 2,
    "content": "// CHECK-LABEL: @updated\nutil.func @updated() { ... }"
  },
  {
    "file": "test/bar.mlir",
    "name": "my_function",
    "content": "..."
  }
]
```

**Extract-edit-replace workflow**:
```bash
# Extract case, edit it, replace it back
iree-lit-extract test.mlir --case 3 > /tmp/case3.mlir
$EDITOR /tmp/case3.mlir
iree-lit-replace test.mlir --case 3 < /tmp/case3.mlir

# Or pipe through transformations
iree-lit-extract test.mlir --case 3 | sed 's/old/new/' | iree-lit-replace test.mlir --case 3
```

### Implementation Details

**Safe atomic writes**:
1. Read original file
2. Parse test cases
3. Replace specified case content
4. Validate case structure
5. Write to temporary file
6. Move original to `.bak`
7. Rename temporary file to original
8. If any step fails, restore from backup

**Case-object-based replacement**:
- Uses `TestCase` objects, not line numbers
- Stable across file edits (case numbers preserved)
- Validates case boundaries (`// -----` delimiters)
- Preserves RUN lines and file structure

**RUN line handling**:
- Strips RUN lines from replacement content by default
- Use `--require-label` to enforce CHECK-LABEL in content
- Validates IR structure before writing (optional)

**Validation** (with `--validate`):
- Runs `iree-opt --verify-diagnostics` on replacement content
- Catches IR errors before writing to file
- Configurable timeout (default 30s)
- Gracefully degrades if iree-opt not found

### Advanced Workflows

**Batch update from extract output**:
```bash
# Extract multiple cases, modify, and replace
iree-lit-extract test.mlir --case 1,2,3 --json > cases.json

# Edit cases.json programmatically (jq, Python, etc.)
jq '.[].content |= gsub("old"; "new")' cases.json > modified.json

# Add file field to each case
jq --arg file "test.mlir" '.[].file = $file' modified.json > ready.json

# Replace all cases
iree-lit-replace --mode json < ready.json
```

**Cross-file operations**:
```bash
# Move case from file A to file B
iree-lit-extract fileA.mlir --case 3 --json | \
  jq '.[0] | .file = "fileB.mlir" | .number = 2' | \
  iree-lit-replace --mode json
```

**Automated test output capture**:
```bash
# Run compiler pass, capture output, update test
iree-opt test.mlir --pass-pipeline | \
  iree-lit-replace test.mlir --case 3 --validate
```

### JSON Mode Details

**Input schema** (array of replacement objects):
```json
[
  {
    "file": "test.mlir",             // Optional if --test-file provided
    "number": 2,                     // XOR with "name" - case number
    "name": "function_name",         // XOR with "number" - case name
    "content": "// CHECK-LABEL...",  // Required - replacement content

    // Optional per-case flags (override global flags):
    "replace_run_lines": true,       // Allow changing RUN lines
    "allow_empty": true,             // Allow empty content
    "require_label": false           // Require CHECK-LABEL in content
  }
]
```

**File resolution rules**:
- If `file` field present in JSON: use it
- Else if `--test-file` CLI argument provided: use CLI file for all replacements
- Else error (must specify file somewhere)

**Unified output schema** (matches text mode with `--json`):
```json
{
  "modified_files": 1,
  "modified_cases": 2,
  "unchanged_cases": 0,
  "dry_run": false,                // true for --dry-run mode
  "file_results": [
    {
      "file": "test.mlir",
      "total_cases": 3,              // Total cases replaced in this file
      "modified": 2,                 // How many were actually changed
      "unchanged": 1,                // How many were skipped (identical)
      "dry_run": false,
      "cases": [
        {
          "number": 1,
          "name": "function_name",
          "changed": true,
          "reason": "content differs"  // Optional explanation
        }
      ],
      "diff": "..."                  // Unified diff (empty if no changes)
    }
  ],
  "errors": [
    {
      "file": "test.mlir",
      "replacement": 1,              // Optional: which replacement entry
      "error": "..."                 // Error message
    }
  ],
  "warnings": [
    {
      "file": "test.mlir",
      "warning": "..."               // Warning message
    }
  ]
}
```

The output schema is consistent across text mode (with `--json`) and JSON batch mode,
both for dry-run and commit operations.

### Troubleshooting Guide

**Problem: "Case not found"**

```
Error: No case found with number 5 in test.mlir (file has 3 cases)
```

**Solution**:
1. List cases: `iree-lit-list test.mlir`
2. Verify case number is within range
3. Or use case name: `--name function_name`

**Problem: "Content validation failed"**

```
Error: Replacement content validation failed:
  iree-opt reported errors:
  test.mlir:10:5: error: expected SSA operand
```

**Solution**:
1. Check replacement content is valid MLIR
2. Run validation manually: `echo "content" | iree-opt --verify-diagnostics`
3. Fix IR errors in replacement content
4. Or skip validation: remove `--validate` flag (not recommended)

**Problem: "Missing CHECK-LABEL"**

```
Error: Replacement content must have CHECK-LABEL matching case name
  Expected label: @my_function
  Found labels: (none)
```

**Solution** (if using `--require-label`):
1. Add CHECK-LABEL to replacement content
2. Ensure label matches expected name
3. Or remove `--require-label` flag

**Problem: "Backup file exists"**

```
Error: Backup file test.mlir.bak already exists
  Remove it manually or use --force to overwrite
```

**Solution**:
1. Check if backup is important: `cat test.mlir.bak`
2. Remove if not needed: `rm test.mlir.bak`
3. Or force overwrite: `iree-lit-replace --force ...` (be careful!)

**Problem: Replacement corrupted file**

**Symptoms**: File syntax is broken after replacement.

**Solution**:
1. Restore from backup: `mv test.mlir.bak test.mlir`
2. Use `--dry-run` to preview changes first
3. Report as bug if replacement corrupted valid input

**Problem: Dry-run diff is hard to read**

**Solution**:
1. Use `--pretty` for colored diff: `iree-lit-replace --dry-run --pretty ...`
2. Or pipe to less with color: `iree-lit-replace --dry-run --pretty ... | less -R`
3. Or use `--json` for programmatic inspection

**Problem: JSON schema validation failed**

```
Error: JSON schema validation failed:
  - Replacement 1: missing 'content' field
  - Replacement 2: invalid 'case' value (must be integer >= 1)
```

**Solution**:
1. Verify JSON schema matches documented format
2. Check all required fields are present
3. Validate JSON syntax: `cat input.json | jq .`
4. Check field types (case is int, content is string)

**Problem: Validation timeout**

```
Warning: Validation timed out after 30s
```

**Solution**:
1. Increase timeout: `--validate-timeout 60`
2. Check if IR is extremely large or complex
3. Or skip validation if timeout persists

**Problem: iree-opt not found**

```
Warning: Skipping validation (iree-opt not found).
Set IREE_BUILD_DIR to enable validation.
```

**Solution**:
1. Build IREE: `cmake --build build --target iree-opt`
2. Set build dir: `export IREE_BUILD_DIR=/path/to/build`
3. Or ignore warning if validation not needed

### Integration with Other Tools

**Extract → Replace workflow**:
```bash
# Extract, edit, replace
iree-lit-extract test.mlir --case 3 | \
  $EDITOR - | \
  iree-lit-replace test.mlir --case 3
```

**List → Extract → Replace workflow**:
```bash
# List all named cases
iree-lit-list test.mlir --names | while read name; do
  # Extract, transform, replace
  iree-lit-extract test.mlir --name "$name" | \
    transform_script | \
    iree-lit-replace test.mlir --name "$name"
done
```

**Replace → Test workflow**:
```bash
# Replace content and validate
iree-lit-replace test.mlir --case 3 --validate < new.mlir
# Then run test
iree-lit-test test.mlir --case 3
```

### Design Principles

**Safety first**:
- Atomic writes with backup files
- Validation before writing (optional)
- Dry-run mode for preview
- Clear error messages

**Case-object stability**:
- Replace by case number/name, not line numbers
- Case numbers preserved across edits
- Robust to whitespace and formatting changes

**Composability**:
- Works with standard Unix pipes
- JSON mode for programmatic use
- Integrates with other iree-lit-* tools

**Token efficiency**:
- Concise default output
- `--quiet` suppresses non-essential messages
- JSON output for LLM consumption

### See Also

- `iree-lit-list` - List test cases
- `iree-lit-extract` - Extract test case content
- `iree-lit-test` - Run tests in isolation
- Implementation: `tools/utils/lit_tools/iree_lit_replace.py`
- Tests: `tools/utils/test/lit_tests/test_replace.py`
