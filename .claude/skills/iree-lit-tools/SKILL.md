---
name: iree-lit-tools
description: MLIR lit test authoring, linting, debugging, and manipulation for IREE
allowed-tools: Bash, Read, Grep, Glob, Edit
---

# IREE Lit Test Tools

Tools for working with MLIR lit tests: listing, extracting, replacing, running, and linting.

## When to Use

- **Writing or fixing lit tests**: Use `iree-lit-lint` to check for style violations, `iree-lit-test` to run isolated cases
- **Debugging a failing test**: Use `iree-lit-list` to see test structure, `iree-lit-extract` to get a specific case, `iree-lit-test --case N` to run it
- **Updating CHECK patterns**: Use `iree-lit-extract` to get the case, run the pipeline manually to see actual output, then `iree-lit-replace` to update
- **Understanding test structure**: Use `iree-lit-list --names` to see what a file tests

## Tool Reference

### iree-lit-list
List test cases in a file.
```bash
iree-lit-list test.mlir              # Show all cases with metadata
iree-lit-list test.mlir --count      # Just the count
iree-lit-list test.mlir --names      # Space-separated names
iree-lit-list test.mlir --json       # Machine-readable
```

### iree-lit-extract
Extract individual test cases.
```bash
iree-lit-extract test.mlir --case 3        # Extract case #3
iree-lit-extract test.mlir --line 123      # Extract case containing line 123
iree-lit-extract test.mlir --name "foo"    # Extract case named "foo"
iree-lit-extract test.mlir -c 1,3,5        # Multiple cases
```

### iree-lit-replace
Replace a test case atomically (reads new content from stdin).
```bash
iree-lit-replace test.mlir --case 3 < fixed_case.mlir

# Heredoc for inline replacement (no temp files needed):
iree-lit-replace test.mlir --case 3 <<'EOF'
// RUN: iree-opt --my-pass %s | FileCheck %s
// CHECK-LABEL: @my_test
func.func @my_test() {
  return
}
EOF
```

### iree-lit-test
Run tests in isolation with debug capabilities.
```bash
iree-lit-test test.mlir                    # Run all cases
iree-lit-test test.mlir --case 2           # Run only case #2
iree-lit-test test.mlir -c 1-3             # Run cases 1, 2, 3
iree-lit-test test.mlir --name "foo"       # Run case named "foo"
iree-lit-test test.mlir --verbose          # Show full output
iree-lit-test test.mlir --extra-flags "--debug"  # Inject flags
iree-lit-test test.mlir --dry-run          # Show commands without running

# Heredoc for rapid testing (no temp files needed):
iree-lit-test --run 'iree-opt --my-pass %s | FileCheck %s' <<'EOF'
// CHECK-LABEL: @test_fusion
// CHECK: my.fused_op
func.func @test_fusion() {
  %0 = my.op1
  %1 = my.op2 %0
  return
}
EOF
```

### iree-lit-lint
Lint tests against the style guide.
```bash
iree-lit-lint test.mlir                    # Lint all cases
iree-lit-lint test.mlir --case 2           # Lint only case #2
iree-lit-lint test.mlir --errors-only      # Only show errors
iree-lit-lint test.mlir --help-style-guide # Show full style guide
```

## Core Style Principles

For the full style guide, run `iree-lit-lint --help-style-guide`.

**1. Verify Transformations, Not Syntax**
Tests must prove the transformation is correct. Capture SSA values the pass touches.
```mlir
// BAD - wildcards the critical value:
// CHECK: stream.async.execute await({{.+}})

// GOOD - captures and verifies:
// CHECK: %[[TIMEPOINT:.+]] = stream.timepoint.join
// CHECK: stream.async.execute await(%[[TIMEPOINT]])
```

**2. Track Data Flow**
Values flowing from producer to consumer must be captured at both ends.
```mlir
// CHECK: %[[SIZE:.+]] = stream.tensor.sizeof
// CHECK: stream.resource.alloc ... {%[[SIZE]]}  ← Same capture
```

**3. Capture Multi-Result Operations Completely**
If an op returns `(resource, timepoint)`, capture both.
```mlir
// CHECK: %[[RESOURCE:.+]], %[[TIMEPOINT:.+]] = stream.async.execute
```

**4. Use Semantic Names**
Match IR names: `%transient_size` → `%[[TRANSIENT_SIZE:.+]]`.

**5. No Raw SSA Identifiers**
Use named captures, not `%0`, `%arg0`, `%buffer`.
```mlir
// BAD:  // CHECK: %5 = arith.addi %0, %arg0
// GOOD: // CHECK: %[[LHS:.+]] = ...
//       // CHECK: %[[RHS:.+]] = ...
//       // CHECK: %[[RESULT:.+]] = arith.addi %[[LHS]], %[[RHS]]
```

**6. Wildcards Only for Structural IR**
Wildcard types, affinities, and debug attrs. Capture everything the pass modifies.

## Rapid Testing with Heredocs

The fastest way to test transformations without temp files:

```bash
# Test a pass inline with heredoc
iree-lit-test --run 'iree-opt --iree-util-fold-globals %s | FileCheck %s' <<'EOF'
// CHECK-LABEL: util.func @test
// CHECK-NOT: util.global
util.global private @unused : i32
util.func @test() {
  util.return
}
EOF

# Replace a test case inline
iree-lit-replace test.mlir --case 2 <<'EOF'
// CHECK-LABEL: @fixed_test
// CHECK: %[[RESULT:.+]] = arith.addi
func.func @fixed_test(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}
EOF

# Lint inline IR
iree-lit-lint <<'EOF'
// RUN: iree-opt %s
// CHECK: arith.addi %arg0, %arg1
func.func @bad_test(%arg0: i32, %arg1: i32) {
  %0 = arith.addi %arg0, %arg1 : i32
  return
}
EOF
```

## Common Workflows

### Debug a Failing Test
```bash
# 1. See test structure
iree-lit-list failing_test.mlir

# 2. Run the failing case in isolation
iree-lit-test failing_test.mlir --case 3 --verbose

# 3. Extract and run manually to see actual output
iree-lit-extract failing_test.mlir --case 3 > /tmp/case.mlir
iree-opt --pass-pipeline="..." /tmp/case.mlir

# 4. Compare expected vs actual, update CHECK patterns
```

### Fix Lint Errors
```bash
# 1. Run linter
iree-lit-lint test.mlir

# 2. See specific issues
iree-lit-lint test.mlir --case 2 --full-json

# 3. Read the style guide for context
iree-lit-lint --help-style-guide | grep -A 20 "Raw SSA"
```

### Update a Test After Pass Changes
```bash
# 1. Extract the case
iree-lit-extract test.mlir --case 2 -o /tmp/case.mlir

# 2. Run pass, capture new output
iree-opt --my-pass /tmp/case.mlir > /tmp/new_output.mlir

# 3. Update CHECK patterns manually or via editor

# 4. Replace the case
cat /tmp/updated_case.mlir | iree-lit-replace test.mlir --case 2

# 5. Verify
iree-lit-test test.mlir --case 2
```

## Environment

Tools auto-detect build directories. Override with:
```bash
export IREE_BUILD_DIR=/path/to/build
```

Add tools to PATH if not found:
```bash
export PATH="$PATH:$IREE_ROOT/tools/utils/bin"
```
