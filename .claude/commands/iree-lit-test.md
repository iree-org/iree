---
allowed-tools: Bash(iree-lit-*), Bash(iree-opt:*), Read, Edit
description: Run and debug a lit test
---

## Context

**Target file(s):**
!`if [ -n "$ARGUMENTS" ]; then echo "$ARGUMENTS"; else git diff --name-only HEAD 2>/dev/null | grep '\.mlir$' | head -5 || echo "No .mlir files in diff"; fi`

**Test structure:**
!`file="$ARGUMENTS"; if [ -z "$file" ]; then file=$(git diff --name-only HEAD 2>/dev/null | grep '\.mlir$' | head -1); fi; if [ -n "$file" ] && [ -f "$file" ]; then PYTHONPATH=tools/utils python3 -m lit_tools.iree_lit_list "$file" 2>&1; else echo "No test file found"; fi`

## Your Task

Run and debug the lit test(s):

1. **If no file specified**: Work with the modified `.mlir` files shown above

2. **Run the test**:
   ```bash
   iree-lit-test <file> --case N          # Run specific case
   iree-lit-test <file> --verbose         # See full output
   iree-lit-test <file> --dry-run         # See commands without running
   ```

3. **If a test fails**:
   - Use `--verbose` to see the actual vs expected output
   - Extract the case: `iree-lit-extract <file> --case N`
   - Run the pipeline manually to see actual IR
   - Update CHECK patterns to match actual output

4. **Before finishing**:
   - Run `iree-lit-lint <file>` to check for style issues
   - Verify all cases pass: `iree-lit-test <file>`

Use the `iree-lit-tools` skill for guidance on style and common workflows.
