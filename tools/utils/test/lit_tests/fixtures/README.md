# Lit Test Fixtures

Test fixtures for validating lit tool functionality. Each fixture tests specific edge cases and scenarios.

## Fixture Catalog

### `simple_test.mlir`
**Purpose**: Single test case without delimiters

**Characteristics**:
- No `// -----` delimiter (single test case file)
- Has `CHECK-LABEL` for named case extraction
- Contains RUN line at beginning
- 1 test case named `@simple_function`

**Used to test**:
- Parsing files with single test case
- Extracting test case by name
- Extracting RUN lines from file header
- Basic CHECK pattern counting

**Structure**:
```
Lines 1-2:  RUN directive
Lines 3-9:  Single test case with CHECK patterns
```

---

### `split_test.mlir`
**Purpose**: Multiple test cases separated by `// -----` delimiters

**Characteristics**:
- Uses `// -----` delimiters to separate cases
- Has multi-line RUN directive at beginning
- 3 test cases, all named with `CHECK-LABEL`
- Each case has different complexity (varying CHECK patterns)

**Test cases**:
1. `@first_case` (lines 5-11) - 2 CHECK patterns
2. `@second_case` (lines 15-23) - 3 CHECK patterns
3. `@third_case` (lines 27-33) - 2 CHECK patterns

**Used to test**:
- Parsing files with multiple test cases
- Splitting on `// -----` delimiter
- Extracting individual test cases by number
- Extracting test cases by name
- Boundary detection (start/end lines)
- Multi-line RUN directive handling

**Structure**:
```
Lines 1-3:   Multi-line RUN directive (with continuations)
Lines 5-11:  Test case 1: @first_case
Line 13:     Delimiter: // -----
Lines 15-23: Test case 2: @second_case
Line 25:     Delimiter: // -----
Lines 27-33: Test case 3: @third_case
```

---

### `names_test.mlir`
**Purpose**: Function names with special characters

**Characteristics**:
- Single test case (no delimiters)
- Function name contains punctuation: `@foo.bar$baz-1`
- Tests CHECK-LABEL extraction with special characters

**Used to test**:
- Parsing function names with dots (`.`)
- Parsing function names with dollar signs (`$`)
- Parsing function names with hyphens (`-`)
- Parsing function names with numbers
- CHECK-LABEL regex robustness

**Structure**:
```
Lines 1-2: RUN directive
Lines 3-7: Single test case with punctuated name
```

---

### `run_variants.mlir`
**Purpose**: RUN line format variations

**Characteristics**:
- Indented comments before RUN lines
- Mixed RUN line formats: `//RUN:` (no space) and `// RUN:` (with space)
- Multi-line RUN directive with continuations
- Tests RUN line extraction with various whitespace patterns

**Used to test**:
- Extracting RUN lines with no space after `//`
- Extracting RUN lines with space after `//`
- Handling indented comments before RUN lines
- Multi-line RUN directive extraction
- RUN line normalization

**Structure**:
```
Lines 1-2: Indented comments (NOT RUN lines)
Lines 3-5: Multi-line RUN directive with mixed formatting
Lines 7-11: Single test case
```

---

### `invalid_ir.mlir`
**Purpose**: Test invalid MLIR that fails verification

**Characteristics**:
- Contains intentionally invalid IR (arith.addi with float operands)
- Uses `--verify-diagnostics` flag
- Demonstrates IR verification error detection
- Uses `expected-error` directive to mark expected failures

**Used to test**:
- Invalid IR error detection in lit_wrapper
- Error message extraction from verification failures
- Diagnostic handling with `--verify-diagnostics`
- Error classification (IR validation vs other errors)

**Structure**:
```
Lines 1:     RUN directive with --verify-diagnostics
Lines 3-6:   Comment explaining the invalid IR
Lines 8-13:  Function with intentionally invalid arith.addi operation
```

**Expected behavior**:
- Test will FAIL with IR verification error when run without `--verify-diagnostics`
- Test will PASS when run with `--verify-diagnostics` (diagnostics are expected)
- Error extractor should detect and classify as "IR verification failed"

---

### `failing_scattered_run_lines.mlir`
**Purpose**: Test cases with RUN lines scattered throughout (edge case)

**Characteristics**:
- Has RUN lines interspersed within test cases (non-standard)
- Multiple delimited test cases
- Each case has its own RUN line (not in file header)
- Tests handling of non-standard RUN line placement

**Used to test**:
- RUN line extraction from within test cases
- Proper association of RUN lines with specific cases
- Handling edge cases in RUN line placement
- Validation of scattered vs header RUN line patterns

**Structure**:
```
Lines vary: RUN lines scattered throughout cases
Multiple test cases with // ----- delimiters
Non-standard but valid lit test format
```

**Expected behavior**:
- Tool should extract case-specific RUN lines correctly
- May require special handling vs standard header RUN lines
- Used to verify robustness of RUN line extraction logic

---

## Fixture Naming Convention

Fixtures follow the pattern: `<scenario>_test.mlir`

**Examples**:
- `simple_test.mlir` - Simple/basic scenario
- `split_test.mlir` - Split file scenario
- `names_test.mlir` - Special names scenario
- `run_variants.mlir` - RUN line variants

## Adding New Fixtures

When adding a new fixture:

1. **Name it descriptively**: `<what_it_tests>_test.mlir`
2. **Document it in this README**: Add a new section explaining:
   - Purpose (one-line summary)
   - Characteristics (what makes it special)
   - Used to test (what scenarios it covers)
   - Structure (line-by-line breakdown)
3. **Keep it minimal**: Only include what's needed to test the scenario
4. **Use real MLIR syntax**: Fixtures should be valid MLIR (even if simplified)

## Test Coverage Matrix

| Scenario | Fixture | Test Cases | Notes |
|----------|---------|-----------|-------|
| Single case file | `simple_test.mlir` | 1 | No delimiters |
| Multiple cases | `split_test.mlir` | 3 | With `// -----` |
| Special characters in names | `names_test.mlir` | 1 | Dots, dollars, hyphens |
| RUN line variations | `run_variants.mlir` | 1 | Mixed formatting |
| Invalid IR | `invalid_ir.mlir` | 1 | Verification failure |
| Scattered RUN lines | `failing_scattered_run_lines.mlir` | Multiple | RUN lines within cases |

## Missing Coverage (Potential Future Fixtures)

**Suggested additions**:
- `unnamed_cases_test.mlir` - Test cases without CHECK-LABEL (unnamed)
- `large_file_test.mlir` - File with 10+ test cases (stress test)
- `empty_case_test.mlir` - Test case with no content between delimiters
- `nested_labels_test.mlir` - Multiple CHECK-LABELs in one case
- `no_checks_test.mlir` - Test case with no CHECK patterns at all

These are not currently needed but could be added if new edge cases are discovered.
