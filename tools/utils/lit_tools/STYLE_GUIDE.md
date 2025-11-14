# MLIR Lit Test Style Guide

This document defines best practices for authoring MLIR lit tests in IREE. These rules ensure tests are robust, maintainable, and actually verify transformations.

**Automated checks** are marked with ✅ and enforced by `iree-lit-lint`.
**Manual review guidelines** are marked with 📖 and should be verified during code review.

---

## Table of Contents

1. [Core Principles for Transformation Verification](#core-principles-for-transformation-verification)
2. [Automated Checks (Errors)](#automated-checks-errors)
3. [Automated Checks (Warnings)](#automated-checks-warnings)
4. [Manual Review Guidelines](#manual-review-guidelines)
5. [Running the Linter](#running-the-linter)

---

## Core Principles for Transformation Verification

This section establishes the foundational principles that inform all specific rules below. These principles answer the question: **"What makes a lit test verify correctness, not just syntax?"**

Understanding these principles helps you write robust tests that catch bugs and remain maintainable as code evolves.

---

### 1. 📖 Transform-Aware Capture Rule

**Rule**: If an SSA value is transformed by the pass under test (or intentionally NOT transformed when you'd expect it to be), that value MUST be captured and verified at all relevant sites.

**Why**: The test verifies the TRANSFORMATION, not just that IR parses. Uncaptured values that the pass touches represent gaps in test coverage. When transformations fail, tests without comprehensive captures may still pass, hiding bugs.

**Examples**:

```mlir
✅ GOOD - Captures both resource and timepoint results:
// CHECK: %[[R:.+]], %[[TP:.+]] = stream.async.execute
// CHECK: stream.async.execute await(%[[TP]]) =>

❌ BAD - Missing timepoint capture (incomplete verification):
// CHECK: %[[R:.+]] = stream.async.execute
// (Timepoint transformation not verified!)

✅ GOOD - Verifies the specific operand after transformation:
// CHECK: %[[JOINED:.+]] = stream.timepoint.join
// CHECK: stream.timepoint.await %[[JOINED]]

❌ BAD - Wildcards the critical operand (hides verification):
// CHECK: stream.timepoint.await {{.+}}
// (Which timepoint? Transformation hidden!)
```

**When to Ignore**: Never. This is the foundational principle. If you're not verifying transformations, the test provides no value.

---

### 2. 📖 Critical Feature Verification Rule

**Rule**: If a pass's primary purpose is to transform, optimize, or analyze a specific *feature* or *property* of the IR (e.g., memory access patterns, control flow structures, dialect-specific types like timepoints, attribute propagation), then all relevant SSA values and operations related to that feature MUST be comprehensively captured and verified. Avoid wildcards for these critical elements where their specific properties are under test.

**Why**: When a pass exists specifically to handle a particular IR feature, that feature IS the test. Wildcarding critical elements undermines the entire purpose of the test. This is a generalization of domain-specific requirements: timepoints for Stream dialect, memory descriptors for HAL, control flow for SCF canonicalization, etc.

**Examples**:

```mlir
✅ GOOD - Stream timepoint pass captures ALL timepoints:
// CHECK: %[[TP1:.+]] = stream.async.execute
// CHECK: %[[TP2:.+]] = stream.async.execute
// CHECK: %[[JOINED:.+]] = stream.timepoint.join max(%[[TP1]], %[[TP2]])
// CHECK: stream.async.execute await(%[[JOINED]]) =>

❌ BAD - Wildcarding critical feature:
// CHECK: stream.async.execute await({{.+}}) =>
// (Testing timepoint elision but not verifying which timepoint!)

✅ GOOD - Memory aliasing pass captures ALL buffer descriptors:
// CHECK: %[[BUF1:.+]] = hal.buffer.subspan<%[[BASE:.+]]
// CHECK: %[[BUF2:.+]] = hal.buffer.subspan<%[[BASE]]
// (Proves both buffers alias the same base)

❌ BAD - Missing descriptor relationships:
// CHECK: hal.buffer.subspan<{{.+}}
// (Can't verify aliasing without tracking base buffer)

✅ GOOD - Attribute propagation pass verifies attributes:
// CHECK: util.global @tensor {{.*}} {noinline}
// (Verifies attribute was propagated)
```

**Domain-Specific Examples**:
- **Stream timepoint passes**: Always capture timepoint SSA values
- **HAL memory passes**: Always capture buffer descriptors and memory types
- **SCF canonicalization**: Always capture control flow results and branch targets
- **Attribute propagation**: Always verify attribute presence/changes

**When to Ignore**: Never for the critical feature being tested. Wildcards are acceptable for other structural IR that the pass doesn't modify.

---

### 3. 📖 Data Flow Verification Rule

**Rule**: If a value flows from producer to consumer across CHECK lines, both the producer's result and consumer's operand must capture/verify the same SSA name reference.

**Why**: This proves data flow correctness. The test must show that value X produced here is value X consumed there. Without tracking the full data flow path, you can't verify the transformation preserved or correctly modified the value's usage.

**Examples**:

```mlir
✅ GOOD - Data flow tracked through captures:
// CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.async.dispatch
// CHECK: %[[JOINED:.+]] = stream.timepoint.join max(%[[TP]], %[[TP2]])
// CHECK: stream.async.dispatch await(%[[JOINED]]) =>
// ^^^ Proves the joined timepoint flows into the await

❌ BAD - Producer captured but consumer wildcarded:
// CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.async.dispatch
// CHECK: stream.async.dispatch await({{.+}}) =>
// ^^^ Lost track of what timepoint is being awaited!

✅ GOOD - Multi-hop data flow:
// CHECK: %[[BASE:.+]] = hal.buffer.allocate
// CHECK: %[[VIEW1:.+]] = hal.buffer.subspan<%[[BASE]]
// CHECK: %[[VIEW2:.+]] = hal.buffer.subspan<%[[BASE]]
// CHECK: hal.device.queue.execute commands([
// CHECK:   hal.command.copy source(%[[VIEW1]]) target(%[[VIEW2]])
// (Proves both views alias same base, copy is from base to itself)

❌ BAD - Broken data flow chain:
// CHECK: %[[BASE:.+]] = hal.buffer.allocate
// CHECK: hal.buffer.subspan<{{.+}}
// CHECK: hal.command.copy source({{.+}}) target({{.+}})
// (Can't verify aliasing or copy semantics!)
```

**When to Ignore**: When the data flow is intentionally abstracted (e.g., testing that ANY constant flows through, not a specific one), but document this clearly in comments.

---

### 4. 📖 Multi-Result Operation Rule

**Rule**: For operations that return multiple results (e.g., `(resource, timepoint)` pairs, tuple unpacking), capture ALL results if ANY result is transformed or verified.

**Why**: Partial capture suggests incomplete understanding. If you care about one result, you should verify both to ensure the operation is correct. Many MLIR operations return structured results where all components matter for correctness.

**Examples**:

```mlir
✅ GOOD - Both results captured:
// CHECK: %[[R:.+]], %[[TP:.+]] = stream.async.dispatch
// CHECK: stream.async.dispatch await(%[[TP]]) => with(%[[R]])

❌ BAD - Only one result captured (half-verified):
// CHECK: %[[R:.+]] = stream.async.dispatch
// (What happened to the timepoint? Was it transformed?)

✅ ACCEPTABLE - Neither captured if op is structural (not transformed):
// CHECK: stream.async.dispatch
// (Pass doesn't touch this dispatch, structural verification only)

✅ GOOD - Tuple unpacking verified completely:
// CHECK: %[[A:.+]], %[[B:.+]] = util.optimization_barrier %[[X]], %[[Y]]
// CHECK: util.return %[[A]], %[[B]]
// (Proves both values flow through correctly)

❌ BAD - Partial tuple verification:
// CHECK: %[[A:.+]] = util.optimization_barrier
// CHECK: util.return {{.+}}, {{.+}}
// (Missing B result, can't verify second value's flow)
```

**When to Ignore**: When the pass genuinely only transforms one result and the other is guaranteed unchanged by design, but document this assumption clearly.

---

### 5. 📖 Control Flow Result Rule

**Rule**: Capture control flow operation results (scf.if, scf.for, scf.while) if:
- The result is used in subsequent operations being verified, OR
- The result type/count is transformed by the pass, OR
- You need to verify data flow through the control flow boundary

Otherwise, wildcards are acceptable for structural control flow.

**Why**: Control flow results define data flow across region boundaries. If your pass modifies what flows out of a loop or conditional, you must verify this. However, if the control flow is purely structural (testing other transformations), wildcarding results reduces noise.

**Examples**:

```mlir
✅ GOOD - Result used later, so captured:
// CHECK: %[[IF_RESULT:.+]]:2 = scf.if
// CHECK:   scf.yield %[[A]], %[[B]]
// CHECK: stream.async.dispatch with(%[[IF_RESULT]]#0, %[[IF_RESULT]]#1)

❌ BAD - Result used but not captured:
// CHECK: scf.if %{{.+}} -> (!stream.resource<external>, !stream.timepoint)
// CHECK: stream.async.dispatch with({{.+}}, {{.+}})
// ^^^ Lost provenance - which values are these?

✅ GOOD - Type transformation verified:
// CHECK: %[[LOOP:.+]] = scf.for {{.+}} -> (!stream.resource<transient>) {
// (Pass changed result from <external> to <transient>)

✅ ACCEPTABLE - Result not used later, structural verification:
// CHECK: scf.for %{{.+}} = %c0 to %c10
// (Testing loop structure, not what flows out)

✅ GOOD - Verifying iter_args data flow:
// CHECK: %[[INIT:.+]] = stream.async.execute
// CHECK: %[[LOOP_RESULT:.+]] = scf.for %{{.+}} = %c0 to %c10
// CHECK-SAME: iter_args(%[[ITER:.+]] = %[[INIT]])
// CHECK:   scf.yield %[[ITER]]
// CHECK: util.return %[[LOOP_RESULT]]
// (Proves value flows: init → iter_args → yield → result → return)
```

**When to Ignore**: When control flow is purely structural scaffolding and your pass operates on the operations INSIDE the region, not the region results.

---

### 6. 📖 Attribute and Property Verification Rule

**Rule**: If a pass transforms or propagates attributes, properties (e.g., `readonly`, `noinline`), or metadata on operations or types, those changes MUST be verified explicitly even if the SSA values themselves remain unchanged.

**Why**: Many passes operate purely on attributes or properties without altering SSA values (e.g., aliasing analysis, memory effects, target-specific annotations, inlining decisions). Verifying these changes is as crucial as verifying SSA value transformations. A pass that fails to propagate an attribute correctly can cause miscompilation even if all SSA values are correct.

**Examples**:

```mlir
✅ GOOD - Verifies noinline attribute propagation:
// RUN: iree-opt --iree-util-propagate-function-attributes %s
// CHECK: util.func private @callee {{.*}} attributes {noinline}

❌ BAD - Missing attribute verification:
// CHECK: util.func private @callee
// (Did the pass add noinline? Test doesn't verify!)

✅ GOOD - Verifies memory effect attribute:
// CHECK: util.global @buffer {{.*}} : !util.buffer {nosideeffect}

✅ GOOD - Verifies type attributes:
// CHECK: tensor<4x4xf32, #iree_encoding.encoding<operand_index = 0>>
// (Pass propagated encoding attribute to tensor type)

❌ BAD - Wildcards critical attributes:
// CHECK: util.global @buffer {{.*}} : !util.buffer
// (Missing nosideeffect verification!)

✅ GOOD - Verifies target-specific annotations:
// CHECK: hal.executable.variant @cuda target(<"cuda", ...>) {
// CHECK:   hal.executable.export @dispatch {{.*}} {workgroup_size = [64 : index, 1 : index, 1 : index]}
// (Verifies workgroup size attribute set by pass)
```

**When to Ignore**: When attributes are boilerplate from parsing and unrelated to the transformation (e.g., source location attributes when testing arithmetic canonicalization).

---

### 7. 📖 Wildcard Usage Guidelines

**Rule**: Wildcards (`{{.+}}`) are acceptable ONLY for IR elements that are:
- NOT under test by this specific pass
- Boilerplate or structural scaffolding
- Irrelevant to the transformation being verified

Avoid wildcards for any SSA value, operation, or attribute whose specific form, type, or value is expected to change or is critical to the correctness of the transformation. **When in doubt, capture and verify.**

**Why**: Wildcards hide bugs. Every wildcard is a statement: "I don't care about this value." If you DO care (because your pass touches it), wildcarding means your test won't catch bugs. Overuse of wildcards creates tests that pass for the wrong reasons.

**Examples**:

```mlir
✅ GOOD - Wildcards for structural types the pass doesn't modify:
// RUN: iree-opt --iree-stream-pack-dispatch-operands %s
// CHECK: stream.async.dispatch @dispatch(%[[ARG:.+]]) : ({{.+}}) -> {{.+}}
// (Pass doesn't modify types, only packs operands - types are structural)

❌ BAD - Wildcarding values the pass transforms:
// RUN: iree-opt --iree-stream-elide-timepoints %s
// CHECK: stream.async.execute await({{.+}}) =>
// (Pass specifically transforms timepoint operands - MUST capture!)

✅ GOOD - Wildcard for device affinity (orthogonal to transformation):
// CHECK: stream.async.dispatch @dispatch affinity({{.+}})
// (Testing dispatch packing, not affinity assignment)

❌ BAD - Over-wildcarding masks transformation:
// CHECK: stream.async.dispatch {{.+}}({{.+}}, {{.+}}) : ({{.+}}) -> {{.+}}
// (What dispatch? What operands? Test verifies nothing!)

✅ GOOD - Explicit captures for critical path:
// CHECK: %[[SIZE:.+]] = stream.tensor.sizeof %[[TENSOR:.+]]
// CHECK: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<*>{%[[SIZE]]}
// CHECK: stream.async.dispatch with(%[[ALLOC]])
// (Data flow from size → alloc → dispatch verified)
```

**General Heuristic**:
- Capture: SSA values, critical attributes, values the pass modifies
- Wildcard: Types (unless type transformation), device affinities, debug attributes, structural constants
- When unsure: Capture it. The worst case is an unused capture warning, not a hidden bug.

---

### 8. 📖 Error and Failure Mode Verification

**Rule**: If a pass is expected to diagnose or fail on specific malformed IR, the test MUST verify the expected error message or failure mode using:
- `CHECK-NOT` for unexpected errors (ensuring pass doesn't emit spurious diagnostics)
- `CHECK: error: <expected message>` for specific diagnostics

**Why**: Compiler passes often include validation. Testing that they correctly *reject* invalid input or emit specific diagnostics is as important as testing correct transformations. A pass that accepts invalid IR can lead to downstream failures or miscompilation.

**Examples**:

```mlir
✅ GOOD - Verifies expected error diagnostic:
// RUN: iree-opt --iree-stream-verify-async-access %s -verify-diagnostics
// expected-error @+1 {{resource used before ready}}
%use = stream.async.dispatch with(%resource)

✅ GOOD - Verifies pass rejects malformed IR:
// RUN: not iree-opt --iree-hal-materialize-interfaces %s
// CHECK: error: expected executable source to have public functions

❌ BAD - Missing error verification:
// RUN: not iree-opt --some-pass %s
// (What error should occur? Test doesn't verify specific failure mode!)

✅ GOOD - Verifies no spurious errors on valid input:
// RUN: iree-opt --iree-stream-schedule-allocation %s | FileCheck %s
// CHECK-NOT: error:
// CHECK-NOT: warning:
// CHECK: stream.resource.alloc

✅ GOOD - Verifies specific warning message:
// RUN: iree-opt --iree-util-optimize-ir %s | FileCheck %s --check-prefix=WARNING
// WARNING: warning: potentially infinite loop detected
```

**When to Ignore**: For passes that don't perform validation and assume valid input (e.g., pure optimization passes with well-defined preconditions).

---

### 9. 📖 Semantic Naming Rule

**Rule**: CHECK capture names should match the IR's SSA variable names when possible, respecting case conventions: `%foo_bar` in IR → `%[[FOO_BAR]]` in CHECK patterns.

**Why**: Matched names maintain clarity and make tests easier to maintain. When IR evolves (e.g., `%size` → `%transient_size`), mismatched capture names become misleading. Semantic consistency between IR and CHECK patterns helps reviewers understand what's being tested.

**Examples**:

```mlir
IR: %transient_size = stream.resource.size
✅ GOOD: // CHECK: %[[TRANSIENT_SIZE:.+]] = stream.resource.size
⚠️  ACCEPTABLE: // CHECK: %[[SIZE:.+]] = stream.resource.size
❌ BAD: // CHECK: %[[FOO:.+]] = stream.resource.size (misleading name)

IR: %awaited = stream.timepoint.await
✅ GOOD: // CHECK: %[[AWAITED:.+]] = stream.timepoint.await
❌ BAD: // CHECK: %[[READY:.+]] = stream.timepoint.await (suggests different semantics)

IR: %loop_result = scf.for
✅ GOOD: // CHECK: %[[LOOP_RESULT:.+]] = scf.for
⚠️  ACCEPTABLE: // CHECK: %[[RESULT:.+]] = scf.for (less precise but not misleading)
```

**When Names Mismatch Acceptably**:
- **Dialect conversions**: `%stream_resource` → `%[[TENSOR]]` during lowering
- **Intentional abstractions**: `%0` → `%[[RESULT]]` for generic patterns
- **Consistent abbreviations**: `%awaited_resource` → `%[[AWAITED]]` if used throughout test

**When to Ignore**: When the IR uses non-semantic names like `%0`, use descriptive capture names. The goal is clarity, not blind matching.

---

## Summary of Core Principles

These principles form a hierarchy:

**Foundation**:
1. Verify transformations (Rule 1)
2. Comprehensive coverage for critical features (Rule 2)

**Application**:
3. Track data flow (Rule 3)
4. Capture multi-result operations completely (Rule 4)
5. Verify control flow results when relevant (Rule 5)
6. Verify attribute/property changes (Rule 6)

**Discipline**:
7. Minimize wildcards (Rule 7)
8. Test error paths (Rule 8)
9. Use semantic names (Rule 9)

**Golden Rule**: The test must prove the transformation is correct by explicitly verifying all values, attributes, and properties the pass touches. Wildcards hide bugs.

---

## Automated Checks (Errors)

These violations will cause `iree-lit-lint` to exit with an error code. They represent CRITICAL requirements or common mistakes that break tests.

### 1. ✅ Raw SSA Identifiers in CHECK Lines

**Rule**: CHECK lines must use named captures for ALL SSA values, not raw identifiers like `%0`, `%arg0`, or semantic names like `%buffer`.

**Why**: Raw SSA identifiers are fragile—they change when unrelated code is added or removed, arguments are reordered, or transformations rename values. Named captures verify data flow and make tests resilient to IR structure changes.

**Temporary Exception - MLIR Constants (Anti-Pattern)**: Raw constants like `%c0`, `%c123`, `%c4_i32`, `%cst`, and `%cst_0` are currently exempted from this rule due to widespread usage in existing tests. However, **this is an anti-pattern**:

- `%c123` tells you nothing about what the value represents
- Is it a fill pattern? An offset? An allocation size? A tensor dimension? Which dimension?
- When constants change, the test silently breaks or passes incorrectly
- Prefer semantic captures like `%[[OFFSET:.+]]`, `%[[SIZE:.+]]`, or `%[[FILL_PATTERN:.+]]` that describe purpose

**Examples**:

```mlir
❌ BAD:
  %c0 = arith.constant 0 : index
  // CHECK: %c0 = arith.constant 0

✅ GOOD:
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0

❌ BAD:
  %0 = arith.addf %arg0, %arg0
  // CHECK: %0 = arith.addf

✅ GOOD:
  %0 = arith.addf %arg0, %arg0
  // CHECK: %[[RESULT:.+]] = arith.addf

❌ BAD (operands need captures too):
  // CHECK: arith.addi %arg0, %arg1

✅ GOOD:
  // CHECK: arith.addi %[[LHS:.+]], %[[RHS:.+]]

❌ BAD (semantic names are still raw SSA):
  // CHECK: call @foo(%buffer, %offset)

✅ GOOD:
  // CHECK: call @foo(%[[BUF:.+]], %[[OFF:.+]])
```

**NOLINT Escape Hatch**: For crash reproducers or special cases where raw SSA is intentional, add `// NOLINT` on a preceding line. This suppresses `raw_ssa_identifier` errors for the rest of the test case.

```mlir
// NOLINT: raw_ssa_identifier - crash reproducer needs exact SSA names
// CHECK: %0 = crash.op
// CHECK: return %0
```

**Capturing Function Arguments**: When capturing function arguments in CHECK-SAME lines, use the `[^:]+` pattern (matches until type specifier `:`) instead of `.+` which is greedy:

```mlir
❌ BAD (.+ is greedy - matches too much):
  // CHECK-LABEL: util.func @test(
  // CHECK-SAME: %[[A:.+]]: i32, %[[B:.+]]: i32
  // (A matches "%arg0: i32, %arg1" - entire rest of signature!)

✅ GOOD ([^:]+ stops at colon):
  // CHECK-LABEL: util.func @test(
  // CHECK-SAME: %[[A:[^:]+]]: i32
  // CHECK-SAME: %[[B:[^:]+]]: i32
  // CHECK-SAME: %[[C:[^:]+]]: tensor<4xf32>
  util.func @test(%arg0: i32, %arg1: i32, %arg2: tensor<4xf32>) {
    // CHECK: arith.addi %[[A]], %[[B]]
    %0 = arith.addi %arg0, %arg1 : i32
    ...
  }
```

The `[^:]+` pattern is preferred because:
- **Terse**: Only 5 characters vs 10 for `[a-z0-9_]+`
- **Robust**: Naturally stops at type specifier (`:`)
- **Works with CHECK-SAME**: Doesn't greedily consume commas or subsequent args

---

### 2. ✅ Zero CHECK Lines

**Rule**: Test cases with IR must have CHECK lines to verify behavior.

**Why**: A test with IR but no verification only confirms the IR parses, not that your pass transforms it correctly. Without CHECK lines, the test provides no value—bugs slip through undetected.

**Examples**:

```mlir
❌ BAD:
  // RUN: iree-opt --some-pass %s | FileCheck %s
  util.func @test(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
    util.return %0 : tensor<4xf32>
  }
  (no CHECK lines!)

✅ GOOD:
  // RUN: iree-opt --some-pass %s | FileCheck %s
  // CHECK-LABEL: @test
  util.func @test(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: %[[RESULT:.+]] = arith.addf
    %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
    // CHECK: util.return %[[RESULT]]
    util.return %0 : tensor<4xf32>
  }
```

---

### 3. ✅ TODO Without Explanation

**Rule**: TODO/FIXME/NOTE comments must explain WHY and WHAT needs to be done, not just mark a location.

**Why**: Bare TODO markers provide no context for future readers and will be ignored. Explain the issue, blockers, or planned work.

**Examples**:

```mlir
❌ BAD:
  // TODO
  // TODO: fix this
  // FIXME: broken

✅ GOOD:
  // TODO(#1234): Add support for scf.while after MutableRegionBranchOpInterface lands.
  // NOTE: This await cannot be eliminated because public functions can't return timepoints.
  // FIXME: Dominance check fails for block arguments in loop headers.
```

---

### 4. ✅ CHECK-NOT Without Anchors

**Rule**: CHECK-NOT should be bracketed by positive CHECK lines to define the scope where something shouldn't appear.

**Why**: Without anchors, CHECK-NOT matches too broadly (entire file) or too narrowly (just next line), causing fragile tests.

**Examples**:

```mlir
❌ BAD:
  // CHECK-LABEL: @test
  // CHECK-NOT: stream.timepoint.await
  (what scope? until EOF?)

✅ GOOD:
  // CHECK-LABEL: @test
  // CHECK: scf.if
  // CHECK-NOT: stream.timepoint.await
  // CHECK: scf.yield
  (clearly checks between scf.if and scf.yield)
```

---

### 5. ✅ Clean Separator Lines and Test Identification

**Rule**: Separator lines (`// -----`) must contain ONLY dashes - no extra text, labels, or ordinals. Test cases are identified by CHECK-LABEL or --list ordinals, never by inline comments.

**Why**: Extra text on separators confuses test identification tools and creates maintenance burden when tests are reordered. Test case ordinals change frequently as tests are added/removed, making hardcoded references immediately stale.

**Examples**:

```mlir
❌ BAD (separator with extra text):
  // ----- (Case 2: Zero CHECK lines)
  // ===== TIER 2: WARNINGS =====

✅ GOOD (clean separator):
  // -----

❌ BAD (hardcoded ordinals in comments):
  // Case 3: Tests that TODO explanations work.

✅ GOOD (no ordinals, descriptive intent):
  // Tests that TODO explanations are required.
```

**Spacing Rule**: Add one blank line between test intent comment and first line of test code:

```mlir
✅ GOOD:
  // Tests that raw SSA identifiers in CHECK lines trigger errors.

  // CHECK-LABEL: @raw_ssa_bad
  util.func @raw_ssa_bad() { ... }

❌ BAD (no blank line):
  // Tests that raw SSA identifiers in CHECK lines trigger errors.
  // CHECK-LABEL: @raw_ssa_bad
```

**Test Identification**: Reference tests by:
- **CHECK-LABEL name**: "@raw_ssa_bad"
- **--list ordinal**: "case 3" (from `iree-lit-lint test.mlir --list`)
- **Never**: "Case 3" in comments (ordinals change!)

---

## Automated Checks (Warnings)

These violations generate warnings but don't block commits. They represent best practices that improve test quality.

### 6. ✅ Excessive Wildcards

**Rule**: Avoid more than 2 `{{.+}}` wildcards in a single CHECK line.

**Why**: Wildcards are appropriate for structural IR (types, attributes) that your pass doesn't touch, but overuse masks the actual transformations being verified. Explicit captures ensure the right values flow through operations.

**Examples**:

```mlir
⚠️ WARNING (5 wildcards):
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex::@dispatch[{{.+}}]({{.+}}, {{.+}}, {{.+}}) : ({{.+}})

✅ BETTER:
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex::@dispatch[%[[C0:.+]]](%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (!stream.resource<*>)
```

---

### 7. ✅ Non-Semantic Capture Names

**Rule**: Avoid capture names based on constant values (`%[[C0]]`) or argument positions (`%[[ARG0]]`). Use semantic names that describe purpose.

**Why**: Names like `%[[C0]]` just copy the constant's syntactic name without conveying meaning. If the constant value changes, the capture name becomes misleading. Semantic names describe purpose and remain accurate across code changes.

**Important**: This warning only fires on **capture definitions** (with `:` pattern like `%[[C0:.+]]`), not on **usages** (bare references like `%[[C0]]`). You can safely reference captures throughout your test without triggering additional warnings.

**Examples**:

```mlir
⚠️ WARNING (definition):
  %c0 = arith.constant 0 : index
  // CHECK: %[[C0:.+]] = arith.constant 0

✅ GOOD:
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0

OK (usage, no warning):
  // CHECK: scf.yield %[[OFFSET]]

⚠️ WARNING (definition):
  util.func @test(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
  // CHECK: arith.addf %[[ARG0:.+]], %[[ARG1:.+]]

✅ GOOD:
  util.func @test(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
  // CHECK: arith.addf %[[LHS:.+]], %[[RHS:.+]]

OK (usage, no warning):
  // CHECK: util.return %[[LHS]]
```

---

### 8. ✅ Mismatched Capture Names

**Rule**: When IR contains semantic SSA names like `%transient_size`, CHECK captures should match: `%[[TRANSIENT_SIZE:.+]]` not `%[[SIZE:.+]]`.

**Why**: Mismatched names break the connection between IR and verification, making it unclear what's being tested. Matched names maintain clarity and catch when IR changes.

**Examples**:

```mlir
IR: %transient_size = arith.constant 123 : index

⚠️ WARNING:
  // CHECK: %[[SIZE:.+]] = arith.constant 123

✅ GOOD:
  // CHECK: %[[TRANSIENT_SIZE:.+]] = arith.constant 123

IR: %slice_ready = stream.timepoint.await

⚠️ WARNING:
  // CHECK: %[[READY:.+]] = stream.timepoint.await

✅ GOOD:
  // CHECK: %[[SLICE_READY:.+]] = stream.timepoint.await
```

**Note**: Variance is acceptable for dialect conversion or intentional abbreviations, but exact matches are preferred.

---

### 9. ✅ Wildcards in Terminators

**Rule**: Terminator operations (`scf.yield`, `util.return`, `func.return`, `cf.br`) should verify all operands explicitly, not use wildcards.

**Why**: Terminators define data flow out of regions/blocks—their operands are what transformations modify. Wildcards here mean you're not verifying the transformation worked.

**Examples**:

```mlir
⚠️ WARNING:
  // CHECK: scf.yield {{.+}}, {{.+}}

✅ GOOD:
  // CHECK: scf.yield %[[RESULT]], %[[TP]]

⚠️ WARNING:
  // CHECK: util.return {{.+}}

✅ GOOD:
  // CHECK: util.return %[[AWAITED]]

⚠️ WARNING:
  // CHECK: cf.br ^bb1({{.+}} : i32)

✅ GOOD:
  // CHECK: cf.br ^bb1(%[[VAL]] : i32)
```

---

### 10. ✅ CHECK Without LABEL Context

**Rule**: CHECK lines should appear after a CHECK-LABEL that establishes which function/block is being verified.

**Why**: Without CHECK-LABEL, FileCheck may match lines from the wrong function, causing tests to pass for the wrong reasons.

**Examples**:

```mlir
⚠️ WARNING:
  // CHECK: %[[FOO:.+]] = some.op
  util.func @test() { ... }

✅ GOOD:
  // CHECK-LABEL: @test
  util.func @test() {
  // CHECK: %[[FOO:.+]] = some.op
```

---

### 11. ✅ Unused Captures

**Rule**: Captured values like `%[[FOO:.+]]` should be referenced later in CHECK lines.

**Why**: A capture that's never used suggests incomplete test coverage—you intended to verify something but didn't.

**Examples**:

```mlir
⚠️ WARNING:
  // CHECK: %[[UNUSED:.+]] = stream.async.execute
  // CHECK: util.return %[[OTHER]]

✅ GOOD:
  // CHECK: %[[RESULT:.+]] = stream.async.execute
  // CHECK: util.return %[[RESULT]]

OK (used in CHECK-SAME):
  // CHECK: %[[TP:.+]] = stream.async.execute
  // CHECK-SAME: await(%[[TP]]) =>
```

---

## Manual Review Guidelines

These patterns are too heuristic or context-dependent for reliable automation. Reviewers should verify these during code review.

### 12. 📖 Missing Transformation Verification

**Rule**: When RUN line invokes a transformation pass (not just parsing/printing), verify the transformation occurred.

**Why**: Tests should have CHECK patterns that verify the pass's behavior, not just that IR parses. Look for pass-specific patterns: if running `--elide-timepoints`, verify awaits are removed/folded.

**Examples**:

```mlir
📖 REVIEW NEEDED:
  // RUN: iree-opt --iree-stream-elide-timepoints %s | FileCheck %s
  // CHECK-LABEL: @test
  util.func @test() {
    %tp = stream.timepoint.await ...
    ...
  }
  (no checks for await elimination!)

✅ GOOD:
  // RUN: iree-opt --iree-stream-elide-timepoints %s | FileCheck %s
  // CHECK-LABEL: @test
  // CHECK-NOT: stream.timepoint.await
  util.func @test() { ... }
```

**Manual Review Prompt**: "Pass `X` runs, but I only see checks for unrelated ops. Does this test verify the transformation?"

---

### 13. 📖 Inconsistent Type Wildcards

**Rule**: Type annotations in CHECK lines should be consistent: either verify types explicitly when they're relevant to the transformation, or use wildcards when types are incidental.

**Why**: Mixing styles (some ops with types, some without) suggests unclear test intent. Be systematic about what you're verifying.

**Examples**:

```mlir
📖 INCONSISTENT:
  // CHECK: %[[A:.+]] = some.op : i32
  // CHECK: %[[B:.+]] = other.op
  // CHECK: %[[C:.+]] = third.op : f32

✅ GOOD (types not relevant):
  // CHECK: %[[A:.+]] = some.op
  // CHECK: %[[B:.+]] = other.op
  // CHECK: %[[C:.+]] = third.op

✅ GOOD (types are relevant):
  // CHECK: %[[A:.+]] = some.op : i32
  // CHECK: %[[B:.+]] = other.op : i32
  // CHECK: %[[C:.+]] = third.op : f32
```

**Manual Review Prompt**: "This test wildcards all types. Should it verify the type transformation?"

---

### 14. 📖 Full-Line Comment Punctuation

**Rule**: Full-line comments should be complete sentences ending with proper punctuation (`.`, `!`, `?`).

**Why**: Complete sentences with punctuation maintain professional code quality and make comments easier to read. This ensures comments are thoughtful explanations, not hastily written fragments.

**Examples**:

```mlir
📖 REVIEW NEEDED:
  // This await is redundant
  // Verify that the transformation works

✅ GOOD:
  // This await is redundant and will be eliminated by the pass.
  // Verify that the transformation correctly removes all awaits in this scope.

OK (trailing comment, no punctuation required):
  %foo = bar  // redundant await
```

**Special Considerations**:
- **Multi-line comments**: Only the last line of a comment block needs punctuation
- **Trailing comments**: Short comments after code don't need punctuation
- **CHECK/RUN directives**: Excluded (not narrative comments)

**Manual Review Prompt**: "Do the comments form complete sentences? Are multi-line comments properly punctuated at the end?"

---

### 15. 📖 "Tests that..." Pattern for Test Intent

**Rule**: Test functions and test case comments should clearly state what behavior is being verified, preferably using the pattern "Tests that [behavior]."

**Why**: Clear test intent makes it obvious what the test verifies and easier to debug when it fails. The "Tests that..." pattern forces you to articulate the specific behavior being tested, not just describe the test setup.

**Examples**:

```mlir
📖 REVIEW NEEDED:
  // Await elimination example
  // CHECK-LABEL: @test
  util.func @test() { ... }

✅ GOOD:
  // Tests that await elimination removes redundant awaits before loops.
  // CHECK-LABEL: @test
  util.func @test() { ... }

✅ GOOD (variant):
  // Tests that the pass preserves awaits required for correctness in public functions.
  // CHECK-LABEL: @public_func
  util.func public @public_func() { ... }
```

**Pattern Benefits**:
- Forces you to think: "What am I actually testing?"
- Makes test failures immediately clear: "Tests that X" → if X fails, debug X
- Documents test coverage gaps: if you can't write "Tests that...", the test is unclear
- Helps reviewers quickly understand test purpose

**Manual Review Prompt**: "Can you state what this test verifies in one sentence starting with 'Tests that...'?"

---

## Running the Linter

### Command Line

```bash
# Lint entire file
iree-lit-lint test.mlir

# Lint specific test cases
iree-lit-lint test.mlir --case 2
iree-lit-lint test.mlir --name fold_constants

# Show only errors (suppress warnings)
iree-lit-lint test.mlir --errors-only

# JSON output for tooling
iree-lit-lint test.mlir --json

# JSON with full IR context for LLM processing
iree-lit-lint test.mlir --json --full-json
iree-lit-lint test.mlir --json --full-json --context-lines=10

# View this style guide
iree-lit-lint --help-style-guide
```

### Understanding Warnings

Each warning includes contextual help text with:
- **What the issue is** and why it matters
- **How to fix it** with concrete examples
- **When to ignore** the warning (marked with "Ignore if...")

The "Ignore if..." guidance helps both humans and LLM agents make informed decisions:

```
warning: capture name based on constant value
  // CHECK: %[[C0:.+]] = arith.constant
  help: Use semantic name like %[[OFFSET]] or %[[SIZE]] instead...
  (Ignore if the constant value IS the semantic meaning, e.g.,
   %[[C0]] for zero-initialization pattern.)
```

Common valid reasons to ignore warnings:
- **Non-semantic captures**: When testing argument forwarding or zero-init patterns
- **Excessive wildcards**: When operation has many structural attributes the pass doesn't modify
- **Mismatched names**: When doing dialect conversion (e.g., `%stream_resource` → `%[[TENSOR]]`)
- **Wildcard terminators**: When terminator forwards arguments unchanged and test focuses elsewhere
- **Unused captures**: When capture is for documentation/clarity even though not referenced

### Pre-Commit Integration

The linter runs automatically on modified test files during pre-commit.

### Exit Codes

- **0**: No errors found (warnings may exist)
- **1**: One or more errors found
- **2**: File not found or parse error

---

## Summary

**Key Principles**:
1. **Verify transformations, not syntax** - Tests should prove your pass worked
2. **Use semantic names** - Match IR names, describe purpose
3. **Capture data flow** - Verify operands, especially in terminators
4. **Anchor patterns** - Use CHECK-LABEL and positive CHECKs around CHECK-NOT
5. **Minimize wildcards** - Only wildcard structural IR you don't care about

**When in doubt**:
- Read existing well-tested files in the same directory
- Ask in code review: "Does this test verify what the pass does?"
- Run `iree-lit-lint` early and often
