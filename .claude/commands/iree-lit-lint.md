---
allowed-tools: Bash(iree-lit-lint:*), Bash(iree-lit-*), Read, Edit
description: Lint a lit test for style issues
---

## Context

**Target file(s):**
!`if [ -n "$ARGUMENTS" ]; then echo "$ARGUMENTS"; else git diff --name-only HEAD 2>/dev/null | grep '\.mlir$' | head -5 || echo "No .mlir files in diff"; fi`

**Lint results:**
!`file="$ARGUMENTS"; if [ -z "$file" ]; then files=$(git diff --name-only HEAD 2>/dev/null | grep '\.mlir$'); else files="$file"; fi; for f in $files; do if [ -f "$f" ]; then echo "=== $f ==="; PYTHONPATH=tools/utils python3 -m lit_tools.iree_lit_lint "$f" 2>&1 | head -50; fi; done`

## Your Task

Fix the lint issues shown above:

### Common Fixes

**1. Raw SSA identifiers** (`%0`, `%arg0`, `%buffer`):
```mlir
// BAD:  // CHECK: %5 = arith.addi %0, %arg0
// GOOD: // CHECK: %[[LHS:.+]] = ...
//       // CHECK: %[[RHS:.+]] = ...
//       // CHECK: %[[RESULT:.+]] = arith.addi %[[LHS]], %[[RHS]]
```

**2. Missing CHECK lines**:
Add CHECK patterns that verify the transformation, not just syntax.

**3. Wildcards on critical values**:
```mlir
// BAD:  // CHECK: stream.async.execute await({{.+}})
// GOOD: // CHECK: %[[TIMEPOINT:.+]] = stream.timepoint.join
//       // CHECK: stream.async.execute await(%[[TIMEPOINT]])
```

**4. Incomplete multi-result captures**:
```mlir
// BAD:  // CHECK: %[[RESOURCE:.+]], %{{.+}} = stream.async.execute
// GOOD: // CHECK: %[[RESOURCE:.+]], %[[TIMEPOINT:.+]] = stream.async.execute
```

### After Fixing

1. Re-run lint to verify: `iree-lit-lint <file>`
2. Run the tests to ensure they still pass: `iree-lit-test <file>`

### For More Context

Run `iree-lit-lint --help-style-guide` to see the full style guide with examples.
