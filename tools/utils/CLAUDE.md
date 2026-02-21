# CLAUDE.md - Agent Rules for tools/utils/

Rules for AI agents contributing to IREE developer utilities.

## Critical Rules

1. **Never `print()` directly** - Use `console.error/warn/note/success` for messages
2. **Never `sys.exit(N)`** - Use `return exit_codes.SUCCESS/ERROR/NOT_FOUND`
3. **Never raw file writes** - Use `fs.safe_write_text()` for atomic UTF-8 writes
4. **Always add common flags** - Call `cli.add_common_output_flags(parser)`
5. **JSON purity** - Never mix `print()` with JSON output; use `console.print_json()`

## Test Command

```bash
python -m unittest discover -s tools/utils -p "test_*.py" -v
```

**Expected output**: `Ran 356 tests ... OK`

Never claim tests pass without this output.

## Key Files by Category

### Lit Tools (`lit_tools/`)
- `iree_lit_list.py` - List test cases
- `iree_lit_extract.py` - Extract cases
- `iree_lit_replace.py` - Replace cases
- `iree_lit_test.py` - Run tests
- `iree_lit_lint.py` - Lint tests
- `core/parser.py` - Parse test files
- `core/check_matcher.py` - FileCheck pattern matching
- `STYLE_GUIDE.md` - MLIR test style rules

### CI Tools (`ci/`)
- `iree_ci_triage.py` - Triage CI failures
- `iree_ci_garden.py` - Manage failure corpus
- `core/patterns.py` - Error pattern definitions
- `core/classifier.py` - Pattern matching

### Common (`common/`)
- `console.py` - Output functions (error/warn/note/success)
- `exit_codes.py` - Standard exit codes
- `fs.py` - Safe file operations
- `build_detection.py` - Locate IREE build artifacts

## Import Pattern

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import console, exit_codes, fs
from lit_tools.core import cli
```

Do NOT use relative imports (`from .core import ...`).
Do NOT use inline imports (always keep at the top of the module).

## Full Documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for complete development guidelines.
