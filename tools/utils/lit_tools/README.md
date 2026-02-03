# Lit Test Utilities

Tools for working with MLIR lit test files that use `// -----` delimiters.

## Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `iree-lit-list` | List test cases | `iree-lit-list test.mlir` |
| `iree-lit-extract` | Extract individual cases | `iree-lit-extract test.mlir --case 3` |
| `iree-lit-replace` | Replace case content | `iree-lit-replace test.mlir --case 3 < new.mlir` |
| `iree-lit-test` | Run tests in isolation | `iree-lit-test test.mlir --case 2` |
| `iree-lit-lint` | Lint for style issues | `iree-lit-lint test.mlir` |

## Quick Examples

```bash
# List test cases in a file
iree-lit-list test.mlir

# Run a specific failing case with verbose output
iree-lit-test test.mlir --case 3 --verbose

# Extract a case for manual debugging
iree-lit-extract test.mlir --case 3 > /tmp/case.mlir

# Lint tests for style violations
iree-lit-lint test.mlir

# Replace a case using heredoc (no temp files needed)
iree-lit-replace test.mlir --case 2 <<'EOF'
// CHECK-LABEL: @my_test
// CHECK: %[[RESULT:.+]] = arith.addi
func.func @my_test(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}
EOF

# Test inline IR with heredoc
iree-lit-test --run 'iree-opt --my-pass %s | FileCheck %s' <<'EOF'
// CHECK-LABEL: @test
func.func @test() {
  return
}
EOF
```

## Common Workflow: Debug a Failing Test

```bash
# 1. See test structure
iree-lit-list failing_test.mlir

# 2. Run the failing case in isolation
iree-lit-test failing_test.mlir --case 3 --verbose

# 3. Extract and run manually to see actual output
iree-lit-extract failing_test.mlir --case 3 > /tmp/case.mlir
iree-opt --my-pass /tmp/case.mlir

# 4. Fix and replace
iree-lit-replace failing_test.mlir --case 3 < /tmp/fixed.mlir

# 5. Verify
iree-lit-test failing_test.mlir --case 3
```

## Style Guide

Run `iree-lit-lint --help-style-guide` for the full MLIR test style guide, or see `STYLE_GUIDE.md`.

Key principles:
- Capture SSA values the pass transforms (don't wildcard critical values)
- Track data flow from producer to consumer
- Capture all results of multi-result operations
- Use semantic names matching IR (`%size` → `%[[SIZE:.+]]`)

## Documentation

- `--help` on any tool for detailed usage
- `STYLE_GUIDE.md` - Full MLIR test style guide
- `CONTRIBUTING.md` - Development guide for contributors
