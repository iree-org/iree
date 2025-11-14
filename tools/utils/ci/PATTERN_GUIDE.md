# CI Error Pattern Authoring Guide

This guide explains how to add, modify, and test error patterns for the IREE CI triage system.

---

## Table of Contents

- [Overview](#overview)
- [Pattern Structure](#pattern-structure)
- [Writing Effective Patterns](#writing-effective-patterns)
- [Field Reference](#field-reference)
- [Extraction Rules](#extraction-rules)
- [Testing Patterns](#testing-patterns)
- [Common Pitfalls](#common-pitfalls)
- [Examples](#examples)

---

## Overview

The CI triage system uses **patterns.yaml** to define error signatures. Each pattern:
- Has a regex that matches error text in logs
- Includes metadata (severity, whether it's actionable, description)
- Optionally extracts structured data (file paths, error codes, etc.)

Patterns are used to:
1. **Identify** errors in CI logs
2. **Classify** them by type and severity
3. **Extract** actionable information for fixing
4. **Group** co-occurring errors into root causes

---

## Pattern Structure

### Basic Pattern

```yaml
pattern_name:
  pattern: 'regex pattern here'
  severity: critical|high|medium|low
  actionable: true|false
  context_lines: 5
  description: "Human-readable description"
```

### Pattern with Extraction

```yaml
compile_error:
  pattern: 'error: .*|compilation terminated'
  severity: critical
  actionable: true
  context_lines: 5
  description: "C++ compilation error"
  extract:
    - name: file_path
      regex: '(\S+\.\w+):(\d+):(\d+):'
    - name: error_message
      regex: 'error: (.+)'
```

---

## Writing Effective Patterns

### 1. **Be Specific, Not Generic**

❌ **Bad**: Matches too broadly
```yaml
error_pattern:
  pattern: 'error'  # Matches "error", "errors", "error-free", etc.
```

✅ **Good**: Matches specific error signatures
```yaml
compile_error:
  pattern: 'error: .*|compilation terminated|fatal error:'
  # Matches C++ compiler errors specifically
```

### 2. **Use Anchors When Possible**

❌ **Bad**: Matches anywhere in the line
```yaml
test_pattern:
  pattern: 'failed'  # Matches "prefailed", "failed test", etc.
```

✅ **Good**: Uses word boundaries or specific context
```yaml
test_failed:
  pattern: 'FAILED.*tests?|Test.*failed'
  # Matches "FAILED 3 tests" or "Test foo failed"
```

### 3. **Match Case-Insensitively**

The pattern engine uses `re.IGNORECASE` by default, so you don't need to write:
```yaml
pattern: 'ERROR|Error|error'  # Unnecessary
```

Just write:
```yaml
pattern: 'error:'  # Matches ERROR:, Error:, error:
```

### 4. **Handle Variations**

❌ **Bad**: Only matches one variant
```yaml
rocm_error:
  pattern: 'rocm error'  # Misses "ROCm runtime error"
```

✅ **Good**: Matches all variants
```yaml
rocm_error:
  pattern: 'ROCm error|rocm.*error'
```

### 5. **Escape Special Regex Characters**

Special characters need escaping:
- `.` → `\.` (literal dot)
- `*` → `\*` (literal asterisk)
- `+` → `\+` (literal plus)
- `?` → `\?` (literal question mark)
- `(` → `\(` (literal parenthesis)
- `[` → `\[` (literal bracket)

Example:
```yaml
vk_error:
  pattern: 'VK_ERROR_\w+'  # \w+ matches alphanumeric
```

---

## Field Reference

### `pattern` (required)

**Type**: String (regex)
**Description**: Regular expression to match in log content

**Regex Flags Enabled**:
- `re.IGNORECASE` - Case-insensitive matching
- `re.MULTILINE` - `^` and `$` match line boundaries

**Examples**:
```yaml
# Simple literal match
pattern: 'Segmentation fault'

# Alternation (OR)
pattern: 'error: .*|compilation terminated'

# Character classes
pattern: 'VK_ERROR_\w+'  # Matches VK_ERROR_DEVICE_LOST, VK_ERROR_OUT_OF_MEMORY, etc.

# Wildcards
pattern: 'rocm.*error'  # Matches "rocm error", "rocm runtime error", etc.
```

### `severity` (required)

**Type**: String
**Values**: `critical`, `high`, `medium`, `low`
**Description**: How severe is this error?

**Guidelines**:
- `critical`: Crash, corruption, security issue (segfault, device lost)
- `high`: Compilation error, test failure, assertion
- `medium`: Timeout, OOM, infrastructure issue
- `low`: Network error, download failure

### `actionable` (required)

**Type**: Boolean
**Description**: Can a developer fix this by changing code?

**Guidelines**:
- `true`: Code bug, test failure, compilation error
- `false`: Infrastructure flake, driver crash, network timeout

**Examples**:
```yaml
# actionable: true - fix the undefined reference
undefined_reference:
  pattern: 'undefined reference to'
  actionable: true

# actionable: false - can't fix GPU driver crash in code
cuda_device_lost:
  pattern: 'CUDA.*device.*lost'
  actionable: false
```

### `context_lines` (required)

**Type**: Integer
**Description**: Number of lines to capture before/after match

**Guidelines**:
- Use `3` for simple errors with clear messages
- Use `5-10` for errors needing stack traces or build context
- Use `10+` for segfaults, core dumps (need full stack)

**Example**:
```yaml
compile_error:
  context_lines: 5
  # Captures:
  # - 5 lines before error (may show build commands)
  # - The error line
  # - 5 lines after (may show additional errors)
```

### `description` (required)

**Type**: String
**Description**: Human-readable explanation of what this pattern detects

**Guidelines**:
- Be specific about error type
- Mention the technology if relevant (Vulkan, HIP, Python, etc.)
- Keep it concise (1-2 sentences)

**Examples**:
```yaml
rocclr_memobj:
  description: "ROCm memory object cleanup crash (false CI failure)"

filecheck_failed:
  description: "LLVM FileCheck test failure"

hip_file_not_found:
  description: "HIP file not found (missing device libraries)"
```

### `extract` (optional)

**Type**: List of extraction rules
**Description**: Extract structured data from matches

See [Extraction Rules](#extraction-rules) below for details.

---

## Extraction Rules

Extraction rules pull structured data from error messages for actionable triage.

### Structure

```yaml
extract:
  - name: field_name        # Name for the extracted field
    regex: 'capture pattern' # Regex with capture groups
```

### How It Works

1. The `regex` is matched against the **matched error text** (not the entire log)
2. If the regex has **capture groups** `()`, the groups are extracted
3. Extracted data is stored in `PatternMatch.extracted_fields[field_name]`

### Example: Extract File Path and Line Number

```yaml
compile_error:
  pattern: 'error: .*'
  extract:
    - name: file_path
      regex: '(\S+\.\w+):(\d+):(\d+):'
      # Captures: ('path/to/file.cpp', '145', '12')
    - name: error_message
      regex: 'error: (.+)'
      # Captures: ('use of undeclared identifier')
```

**Input Log**:
```
/home/user/code.cpp:145:12: error: use of undeclared identifier 'foo'
```

**Extracted Fields**:
```python
{
  "file_path": ("/home/user/code.cpp", "145", "12"),
  "error_message": ("use of undeclared identifier 'foo'",)
}
```

### Example: Extract Error Codes

```yaml
hip_error:
  pattern: 'hipError\w+|HIP error'
  extract:
    - name: hip_error_code
      regex: '(hipError\w+)'
      # Captures: ('hipErrorFileNotFound',)
```

### Example: Extract Python Exception Type

```yaml
python_exception:
  pattern: 'Traceback \(most recent call last\)|(?:Error|Exception):'
  extract:
    - name: exception_type
      regex: '(\w+Error|\w+Exception):'
      # Captures: ('ValueError', 'RuntimeError', etc.)
    - name: exception_file
      regex: 'File "([^"]+)", line (\d+)'
      # Captures: ('test.py', '42')
```

---

## Testing Patterns

### 1. **Unit Testing** (Recommended)

Create a test case in `tools/utils/test/ci_tests/test_patterns.py`:

```python
def test_compile_error_pattern():
    """Test that compile_error pattern matches C++ errors."""
    loader = load_default_patterns()
    matcher = PatternMatcher(loader)

    # Test log content
    log_content = """
/home/user/code.cpp:145:12: error: use of undeclared identifier 'foo'
  int x = foo;
          ^
    """

    # Analyze
    results = matcher.analyze_log(log_content)

    # Assertions
    assert 'compile_error' in results
    assert len(results['compile_error']) == 1

    match = results['compile_error'][0]
    assert 'file_path' in match.extracted_fields
    assert match.extracted_fields['file_path'] == ('/home/user/code.cpp', '145', '12')
```

### 2. **Interactive Testing** (Quick Iteration)

```python
# In Python REPL
from ci.core.patterns import load_default_patterns, PatternMatcher

loader = load_default_patterns()
matcher = PatternMatcher(loader)

# Test your pattern
log = "hipErrorFileNotFound: /opt/rocm/amdgcn/lib/bitcode.bc"
results = matcher.analyze_log(log)
print(results)
```

### 3. **Real-World Testing**

```bash
# Analyze actual CI log
iree-ci-triage --run 12345678 --job 987654321
```

---

## Common Pitfalls

### ❌ **Pitfall 1: Pattern Too Broad**

```yaml
# BAD: Matches configuration text, not errors
vulkan_error:
  pattern: 'vulkan'
  # Matches "IREE_VULKAN_DISABLE=1" (false positive)
```

**Fix**: Be specific about error context
```yaml
# GOOD: Matches actual Vulkan API errors
vk_error:
  pattern: 'VK_ERROR_\w+'
  # Only matches VK_ERROR_* codes
```

### ❌ **Pitfall 2: Not Escaping Special Characters**

```yaml
# BAD: . matches any character, not literal dot
file_not_found:
  pattern: 'file.cpp'
  # Matches "file-cpp", "filexcpp", etc.
```

**Fix**: Escape the dot
```yaml
# GOOD: Literal dot
file_not_found:
  pattern: 'file\.cpp'
```

### ❌ **Pitfall 3: Overlapping Patterns**

```yaml
# BAD: Both patterns match the same errors
rocm_error:
  pattern: 'rocm.*error'

rocclr_error:
  pattern: 'rocclr.*error'
  # rocclr errors also match rocm_error!
```

**Fix**: Make patterns mutually exclusive or handle co-occurrence in rules
```yaml
# GOOD: Specific pattern first, generic second
rocclr_memobj:
  pattern: 'Memobj map does not have ptr'
  # Very specific signature

rocm_error:
  pattern: 'ROCm error|rocm.*error'
  # General ROCm errors
```

### ❌ **Pitfall 4: Not Testing Extraction**

```yaml
# Extraction regex doesn't match the pattern context
pattern: 'error occurred'
extract:
  - name: file_path
    regex: '(\S+):(\d+):'
    # Will fail if error message doesn't have "file:line:"
```

**Fix**: Test extraction with realistic error messages

---

## Examples

### Example 1: Simple Error Pattern

**Goal**: Detect Python import errors

```yaml
python_import_error:
  pattern: 'ImportError:|ModuleNotFoundError:'
  severity: high
  actionable: true
  context_lines: 3
  description: "Python import error"
```

**Matches**:
```
ImportError: No module named 'torch'
ModuleNotFoundError: No module named 'iree.runtime'
```

### Example 2: Pattern with Extraction

**Goal**: Detect CUDA errors and extract error codes

```yaml
cuda_error:
  pattern: 'CUDA error|cudaError|CUDA_ERROR'
  severity: high
  actionable: true
  context_lines: 5
  description: "CUDA runtime error"
  extract:
    - name: cuda_error_code
      regex: '(cudaError\w+|CUDA_ERROR_\w+)'
```

**Matches**:
```
CUDA error: cudaErrorMemoryAllocation
CUDA_ERROR_OUT_OF_MEMORY
```

**Extracted**:
```python
{
  "cuda_error_code": ("cudaErrorMemoryAllocation",)
}
```

### Example 3: Complex Pattern with Multiple Extractors

**Goal**: Detect FileCheck failures and extract test file and failed line

```yaml
filecheck_failed:
  pattern: 'FileCheck.*failed'
  severity: high
  actionable: true
  context_lines: 10
  description: "LLVM FileCheck test failure"
  extract:
    - name: check_file
      regex: 'FileCheck.*--check-prefix[es]*=(\S+)'
    - name: failed_line
      regex: '(\S+):(\d+):\d+: error:'
```

**Matches**:
```
FileCheck --check-prefixes=CHECK test.mlir failed
test.mlir:42:10: error: CHECK: expected string not found in input
```

**Extracted**:
```python
{
  "check_file": ("CHECK",),
  "failed_line": ("test.mlir", "42")
}
```

---

## Adding a New Pattern: Step-by-Step

### Step 1: Identify the Error Signature

Find a representative error in a CI log:
```
hipErrorFileNotFound: code object file not found
/opt/rocm-6.0.0/amdgcn/bitcode/oclc.amdgcn.bc
```

### Step 2: Write the Pattern Regex

Start simple:
```yaml
pattern: 'hipErrorFileNotFound'
```

Test if it's too specific or too broad. Adjust:
```yaml
pattern: 'hipErrorFileNotFound'  # Specific to this error code
```

### Step 3: Determine Severity and Actionability

- Is this a code bug or infrastructure? → `actionable: true` (config issue)
- How severe? → `high` (blocks GPU tests)

### Step 4: Choose Context Lines

HIP errors usually need file paths, so capture more context:
```yaml
context_lines: 5
```

### Step 5: Write Extraction Rules (if needed)

Extract the error code:
```yaml
extract:
  - name: hip_error_code
    regex: '(hipError\w+)'
```

### Step 6: Add to patterns.yaml

```yaml
hip_file_not_found:
  pattern: 'hipErrorFileNotFound'
  severity: high
  actionable: true
  context_lines: 5
  description: "HIP file not found (missing device libraries)"
  extract:
    - name: hip_error_code
      regex: '(hipError\w+)'
```

### Step 7: Test It

```python
from ci.core.patterns import load_default_patterns, PatternMatcher

loader = load_default_patterns()
matcher = PatternMatcher(loader)

log = """
hipErrorFileNotFound: code object file not found
/opt/rocm-6.0.0/amdgcn/bitcode/oclc.amdgcn.bc
"""

results = matcher.analyze_log(log)
assert 'hip_file_not_found' in results
print(results['hip_file_not_found'][0].extracted_fields)
# {'hip_error_code': ('hipErrorFileNotFound',)}
```

---

## Pattern Organization

### Group Related Patterns

Organize patterns.yaml by technology/category:

```yaml
patterns:
  # ========================================================================
  # Build and Compilation Errors
  # ========================================================================
  compile_error: ...
  linker_error: ...

  # ========================================================================
  # HIP/ROCm Errors (AMD GPU)
  # ========================================================================
  hip_error: ...
  rocm_error: ...
  rocclr_memobj: ...

  # ========================================================================
  # CUDA Errors (NVIDIA GPU)
  # ========================================================================
  cuda_error: ...
  cuda_oom: ...
```

### Use Descriptive Names

- ✅ `rocclr_memobj` - Specific error type
- ❌ `rocm_error_3` - Meaningless number

### Document Special Cases

```yaml
rocclr_memobj:
  pattern: 'Memobj map does not have ptr'
  severity: critical
  actionable: false  # Infrastructure/driver issue, not code bug
  context_lines: 10
  description: "ROCm memory object cleanup crash (false CI failure)"
  # IMPORTANT: Tests often PASS before this crash during cleanup
  # Creates false negatives - DO NOT fail PRs on this pattern alone
```

---

## Questions?

- Check existing patterns in `patterns.yaml` for examples
- Run pattern tests: `python -m pytest tools/utils/test/ci_tests/test_patterns.py`
- Analyze real logs: `iree-ci-triage --run <run-id>`
