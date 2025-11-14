# Contributing to tools/utils

Development guide for contributors working on IREE developer utilities.

## Pre-commit Hook

This subtree has a scoped pre-commit hook (`utils-validate`):
- Permissions: only `bin/*` are executable; wrappers must have python3 shebang
- Imports: no `from lit ...` (use `from lit_tools ...`)
- JSON purity: never `print()` to stdout from JSON branches

Run manually:
```bash
pre-commit run utils-validate -a
python3 tools/utils/scripts/precommit_utils.py
```

## Code Style

- **Minimal dependencies** - Only essential packages (psutil for lit timeouts); see pyproject.toml
- **argparse for CLI** - Consistent with IREE's existing scripts
- **Apache 2.0 + LLVM header** - Include license header in all files
- **Docstrings with examples** - Module docstring shows usage examples
- **Consistent output modes**:
  - `--json` for machine-readable output
  - `--pretty` for human-friendly formatting (opt-in)
  - default mode should be concise and script-friendly

## Authoring Guidelines (Required)

- Always add common output flags with `cli.add_common_output_flags(parser)`.
- Use `console.error/warn/note/success` for all human messages; never `print()` directly.
- Use `console.print_json(...)` for JSON; do not mix JSON and human text on stdout.
- Honor `--quiet`: suppress non-essential text automatically via `console.*`.
- Return `exit_codes.SUCCESS/ERROR/NOT_FOUND` instead of magic numbers.
- Write files via `fs.safe_write_text(...)` (UTF-8, atomic rename, normalized newlines).
- Shebang policy: only `bin/` wrappers have shebangs.
- **CRITICAL**: Never disable lint checks (ruff, black) with `# noqa` without explicit approval.

### Token Efficiency & LLM-Friendliness

- Default output is concise and stable; opt into `--pretty` for humans.
- `--json` prints only JSON to stdout; human notes go to stderr.
- `--quiet` suppresses notes/warnings/success to save tokens.
- Use shared console/formatting helpers to avoid drift and noise.

## JSON Schema Coordination

- In-tree only: producers and consumers live together; update both in the same PR.
- No version fields: keep payloads small; tests catch breakages.
- For large runs, write JSON to a file (`--json-output`) and post-process with `jq`.

## I/O Output Contract (lit tools)

- Without `-o`:
  - With `--json`: JSON to stdout only
  - Without `--json`: Text to stdout only
- With `-o <file>`:
  - With `--json`: JSON to file; stdout quiet
  - Without `--json`: Text to file; stdout quiet

## Minimal Tool Skeleton

```python
import sys
from pathlib import Path

# Add parent directory to sys.path (required for all tools).
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import console, exit_codes, fs
from lit_tools.core import cli

parser = argparse.ArgumentParser(...)
cli.add_common_output_flags(parser)
args = parser.parse_args()

try:
    # Do work.
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

## Testing Requirements

**Single test command for ALL tests:**

```bash
# From IREE repo root (primary workflow):
python -m unittest discover -s tools/utils -p "test_*.py" -v

# From tools/utils directory:
cd tools/utils
python -m unittest discover -s . -p "test_*.py" -v
```

**Never declare "tests pass" until seeing: `Ran 356 tests ... OK`**

**Key Requirements:**
- unittest for tests (matches IREE's test infrastructure)
- Test coverage >80% for shared libraries
- Fixture-based tests in `test/<category>_tests/fixtures/`
- Cross-platform (Windows, Linux, macOS)
- Python only, no shell scripts

### Fixture-based vs Real-file Testing

**Fixture-based tests** (required for CI):
- Reproducible, controlled content
- Place fixtures in `test/<category>_tests/fixtures/`

**Real-file validation** (development, not CI):
- Manually validate on real IREE test files
- Document in PR description
- Don't add real-file tests to unittest

## Import Strategy (REQUIRED)

### For category tools (lit_tools/, ci/, etc.)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import build_detection, console, exit_codes, fs
from lit_tools.core import cli
```

### For test files

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from lit_tools import iree_lit_list
from common import build_detection
```

### For bin/ wrappers

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lit_tools import iree_lit_list

if __name__ == "__main__":
    sys.exit(iree_lit_list.main(iree_lit_list.parse_arguments()))
```

**CRITICAL: Do NOT use relative imports** (`from .core import ...`). Use absolute imports after sys.path is set up.

## Execution Model

### File naming conventions

- **Python modules**: Use underscores (`iree_lit_list.py`)
- **CLI wrappers**: Use hyphens (`iree-lit-list`)

### Executable permissions

- Only `bin/` wrappers should be executable and have shebangs
- Category modules should NOT have shebangs or be executable

### Module structure

Category modules must include:
```python
if __name__ == "__main__":
    sys.exit(main(parse_arguments()))
```

## Test Organization

### Directory structure

```
tools/utils/test/<category>_tests/test_<module>.py
```

### Naming conventions

| Module Location | Test Location |
|----------------|---------------|
| `lit_tools/iree_lit_list.py` | `test/lit_tests/test_iree_lit_list.py` |
| `lit_tools/core/parser.py` | `test/lit_tests/test_core_parser.py` |
| `common/build_detection.py` | `test/common_tests/test_build_detection.py` |

### Fixtures

Place in `test/<category>_tests/fixtures/`:
```
test/lit_tests/fixtures/
  ├── split_test.mlir          # Multiple cases with // -----
  ├── single_case_test.mlir    # Single case, no delimiters
  └── README.md                # Documents what each fixture tests
```

## Build Detection

### When to use

**Use build_detection when:**
- Tool invokes IREE binaries (`iree-opt`, `iree-compile`, `FileCheck`)
- Tool needs to know build type (Debug vs Release)
- Tool needs to locate LLVM tools

**Skip when:**
- Tool only parses/lists/extracts text
- Tool is a pure utility (formatting, file manipulation)

### Search order

1. `IREE_BUILD_DIR` environment variable (override)
2. `./build/` (in-tree build)
3. `../<worktree>-build/` (worktree pattern)
4. `../iree-build/` (main repo build)

### Error handling patterns

**Optional tool usage:**
```python
try:
    iree_opt = build_detection.find_tool("iree-opt")
except FileNotFoundError:
    console.warn("Skipping validation (iree-opt not found).", args=args)
```

**Required tool:**
```python
try:
    iree_opt = build_detection.find_tool("iree-opt")
except FileNotFoundError as e:
    console.error(f"Cannot find iree-opt.\n{e}", args=args)
    return exit_codes.NOT_FOUND
```

## Adding New Tools

1. Add implementation to appropriate category directory
2. Add comprehensive docstring with examples
3. Write tests (unit + integration) in `test/<category>_tests/`
4. Create a thin Python wrapper in `bin/` (not a symlink)
5. Update category README.md
6. Run full test suite: `python -m unittest discover -s tools/utils -p "test_*.py" -v`

## Category READMEs

- `lit_tools/README.md` - Lit test tools architecture
- `ci/README.md` - CI triage tools
