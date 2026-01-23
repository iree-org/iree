# Lit Test Utilities - Development Guide

Developer documentation for contributors working on lit test tools.

## Architecture

**Core libraries** (`lit_tools/core/`):
- `parser.py` - Parse test files (split by `// -----`, extract cases, RUN lines)
- `check_pattern.py` - FileCheck pattern parsing
- `check_matcher.py` - Pattern matching logic
- `document.py` - Test file AST models
- `verification.py` - IR validation
- `suggestions.py` - Fuzzy name matching
- `cli.py` - Common CLI utilities
- `lit_wrapper.py` - LLVM lit integration

**Tools** (`lit_tools/`):
- `iree_lit_list.py` - List test cases
- `iree_lit_extract.py` - Extract individual cases
- `iree_lit_replace.py` - Replace test cases with new content
- `iree_lit_test.py` - Run tests in isolation
- `iree_lit_lint.py` - Lint tests against style guide

All tools support `--json` for machine-readable output.

## Dependencies

```
lit tools
  ↓
lit_tools/core/* (parser, check_matcher, etc.)
  ↓
common/* (build_detection, console, exit_codes, fs)
  ↓
Python 3 stdlib + psutil
```

## Development Workflow

### Adding a New Tool

1. **Create implementation** in `lit_tools/iree_<tool_name>.py` (use underscores)
2. **Add docstring** with extensive examples (see template below)
3. **Write tests** in `test/lit_tests/test_<tool_name>.py`
4. **Create wrapper** in `bin/iree-<tool-name>` (use hyphens)
5. **Validate**: Run on real IREE test files

### Running Tests

```bash
# Lit-only tests
python3 -m unittest discover -s tools/utils/test/lit_tests -v

# All utils tests
python3 -m unittest discover -s tools/utils -p "test_*.py" -v
```

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
  iree-<category>-<action> input.mlir
  iree-<category>-<action> input.mlir --option value

Examples:
  $ iree-<category>-<action> test.mlir
  [expected output]

Exit codes:
  0 - Success
  1 - Error (invalid input, execution failure, etc.)
  2 - Not found (file doesn't exist, case not found, etc.)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import console, exit_codes, fs
from lit_tools.core import cli
from lit_tools.core.parser import parse_test_file


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
    file_path = Path(args.file)
    if not file_path.exists():
        console.error(f"File not found: {file_path}", args=args)
        return exit_codes.NOT_FOUND

    try:
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

### Binary Wrapper Pattern

Create thin Python wrappers in `bin/` (not symlinks):

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

sys.path.insert(0, str(Path(__file__).parent.parent))

from lit_tools import iree_tool_name

if __name__ == "__main__":
    sys.exit(iree_tool_name.main(iree_tool_name.parse_arguments()))
```

**Naming Convention**:
- Python module: `iree_<tool>_<name>.py` (underscores)
- CLI wrapper: `iree-<tool>-<name>` (hyphens)

**Shebang Policy**:
- ✅ Wrappers in `bin/` MUST have `#!/usr/bin/env python3`
- ❌ Category modules MUST NOT have shebangs

## Code Review Checklist

- [ ] Docstring includes usage examples and exit codes
- [ ] Unit tests added with >80% coverage
- [ ] `--help` output is clear and comprehensive
- [ ] Error messages are descriptive
- [ ] Exit codes follow conventions (0=success, 1=error, 2=not found)
- [ ] Binary wrapper created in `bin/`

## Common Patterns

### Parsing Test Files

```python
from lit_tools.core.parser import parse_test_file

test_file_obj = parse_test_file(Path('test.mlir'))
cases = list(test_file_obj.cases)

case = test_file_obj.find_case_by_number(2)
case = test_file_obj.find_case_by_name('function_name')
case = test_file_obj.find_case_by_line(42)

run_lines = test_file_obj.extract_run_lines()
```

### Build Detection

**When to use**:
- `iree-lit-list` - ❌ No build detection (only parses text)
- `iree-lit-extract` - ⚠️ Optional (only for `--validate` flag)
- `iree-lit-test` - ✅ Required (must run iree-opt and FileCheck)
- `iree-lit-lint` - ❌ No build detection (only parses text)

**Optional validation pattern**:
```python
from common import build_detection

if args.validate:
    try:
        iree_opt = build_detection.find_tool("iree-opt")
        # Validate IR...
    except FileNotFoundError:
        console.warn("Skipping validation (iree-opt not found)", args=args)
```

**Required tool pattern**:
```python
try:
    iree_opt = build_detection.find_tool("iree-opt")
    filecheck = build_detection.find_tool("FileCheck")
except FileNotFoundError as e:
    console.error(f"Cannot run lit tests without IREE build.\n{e}", args=args)
    return exit_codes.NOT_FOUND
```

## Edge Cases

### Multiple CHECK-LABELs in One Case

Parser extracts the **first** CHECK-LABEL as the case name.

### No Functions

Cases without functions show as `(unnamed)`.

### Mixed Named/Unnamed Cases

Files can have both named and unnamed cases - this is supported.

## JSON Output Conventions

**Listing schema**:
```json
{
  "file": "path/to/test.mlir",
  "count": 3,
  "cases": [
    {"number": 1, "name": "foo", "start_line": 1, "end_line": 12, "line_count": 12, "check_count": 3}
  ]
}
```

**Extraction schema** (array of cases):
```json
[
  {
    "number": 2,
    "name": "second_case",
    "start_line": 14,
    "end_line": 24,
    "content": "// CHECK-LABEL: @second_case\nutil.func @second_case() { ... }"
  }
]
```

## Testing Philosophy

- **Unit tests**: Core logic must have >80% coverage
- **Fixture-based tests**: Use `test/lit_tests/fixtures/` for CI
- **Development validation**: Test on real IREE files manually
- **Edge cases**: Test unnamed cases, single case files, large files

**Do NOT** add real-file tests to unittest (files change, breaking CI).

## Tool-Specific Documentation

### iree-lit-test Implementation

Uses LLVM's lit APIs in-process:
1. Build temporary test shard under `/tmp/iree_lit_test_$PID/`
2. Preserve line numbers by prepending `(start_line - 1)` blanks
3. Re-inject RUN lines from header and case body
4. Execute with `lit.discovery` and `lit.run.Run`

### iree-lit-replace Implementation

**Safe atomic writes**:
1. Read original file
2. Parse test cases
3. Replace specified case content
4. Write to temporary file
5. Move original to `.bak`
6. Rename temporary file to original
7. Restore from backup on failure

**Validation features**:
- Name/number consistency checking
- Duplicate case name handling
- Duplicate replacement entry detection

## Troubleshooting

### "Cannot find build directory"

```bash
# Build IREE first
cmake -B build -S . && cmake --build build -j$(nproc)

# Or set environment variable
export IREE_BUILD_DIR=/path/to/build
```

### Test hangs forever

```bash
# Run with shorter timeout
iree-lit-test test.mlir --case 5 --timeout 10

# Or disable timeout and debug
iree-lit-test test.mlir --case 5 --timeout 0 --verbose
```

### Wrong line numbers

This shouldn't happen - iree-lit-test preserves line numbers. Report as bug if seen.

### Case not found by name

```bash
# List cases to verify name
iree-lit-list test.mlir

# Use case number instead
iree-lit-test test.mlir --case 5
```

## See Also

- `STYLE_GUIDE.md` - MLIR test style rules
- `../CONTRIBUTING.md` - Top-level development guidelines
- `../README.md` - User documentation
