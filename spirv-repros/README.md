# IREE amdgcnspirv: LLVM SPIR-V Backend Status

Status of the `amdgcnspirv` (SPIR-V output) mode for IREE's ROCm target backend.
This mode uses the LLVM SPIR-V backend to produce portable SPIR-V binaries that the
HIP runtime JIT-compiles to native ISA at kernel launch.

LLVM version: trunk (as of 2026-03-09, IREE's bundled `third_party/llvm-project`).

## Unsupported IREE Features

The following IREE/ROCDL features are disabled or unavailable in SPIR-V mode:

### Buffer instructions (fat pointers) — biggest codegen difference

IREE's ROCDL backend normally uses buffer fat pointers (AMDGPU address space 7) for
global memory accesses. Buffer descriptors allow hardware bounds checking and can
improve performance by enabling scalar base address + vector offset patterns.

The SPIR-V backend does not support address space 7 — it crashes in
`SPIRVEmitIntrinsics` on `llvm.amdgcn.make.buffer.rsrc`. IREE works around this by
skipping `ROCDLConfigureBufferInstructionsPass` when the target format is
`rocm-spirv-fb`. All memory accesses use regular pointer-based loads/stores instead.

**Impact:** Functional correctness is unaffected. Performance may differ because the
native ISA path uses buffer instructions for all global loads/stores while the SPIR-V
path uses flat/global instructions. The HIP JIT compiler may or may not recover this
via its own optimization.

### i4 integer types

IREE uses `i4` for FP4/INT4 packed formats. SPIR-V requires the `SPV_INTEL_int4`
extension for 4-bit integers, which the AMD HIP JIT does not support.

**Impact:** 2 tests fail at runtime (fp4_f32_conversion, small_float_arith). Fix:
widen i4 to i8 in IREE before SPIR-V emission, or wait for AMD JIT to support i4.

## LLVM SPIR-V Backend Bug Reproducers

Minimal reproducers for bugs in the LLVM SPIR-V backend (`llvm/lib/Target/SPIRV/`).
All reproducers use `llc` from an LLVM build with the SPIR-V target enabled.

## Build llc

```bash
# From an LLVM build with -DLLVM_TARGETS_TO_BUILD="AMDGPU;SPIRV":
cmake --build build --target llc opt llvm-reduce
```

## Crashes (4 bugs)

### Bug 1: `llvm.assume` with operand bundles crashes SPIRVEmitIntrinsics

**File:** `assume_crash.ll`
**Error:** `Assertion 'ArgNo < arg_size() && "Param index out of bounds!"' failed`
**Location:** `llvm/lib/IR/Instructions.cpp:414` (`CallBase::paramHasAttr`)
**Component:** `SPIRVEmitIntrinsics` pass

```bash
llc -mtriple=spirv64-amd-amdhsa assume_crash.ll -o /dev/null
```

**Impact:** Blocks ALL SPIR-V compilation of any LLVM IR containing `llvm.assume` with
operand bundles (e.g., alignment hints). This is extremely common in optimized IR.
IREE workaround: `ROCDLPrepareForSPIRVPass` strips all `llvm.intr.assume` ops.

**Root cause:** `SPIRVEmitIntrinsics::processInstrAfterVisit()` iterates operand bundle
operands as if they were regular call arguments, causing out-of-bounds access on
`paramHasAttr()`.

**Affected tests:** All 25 failing IREE e2e tests hit this first.

---

### Bug 2: SPIRVEmitIntrinsics creates `llvm.spv.gep` with wrong return type for vector-indexed GEPs

**File:** `vector_gep_crash.ll`
**Error:** `Intrinsic has incorrect return type! ptr addrspace(4) @llvm.spv.gep.v1p1.p1`
**Location:** `SPIRVEmitIntrinsics` pass, then `Module Verifier`

```bash
# Requires Bug 1 workaround (strip llvm.assume) to reach this bug.
# This repro file has no llvm.assume, so it triggers directly:
llc -mtriple=spirv64-amd-amdhsa vector_gep_crash.ll -o /dev/null
```

**Root cause:** When a `getelementptr` uses a vector index (`<1 x i64>`), the result
is a vector of pointers (`<1 x ptr addrspace(1)>`). `SPIRVEmitIntrinsics` rewrites this
to `@llvm.spv.gep` but assigns a scalar pointer return type instead of the vector type.

**Affected tests:** 12 IREE tests (pack_i8, fft, fft_complex, gather, map_store,
torch_index_select variants, gather_like_ops, unpack)

---

### Bug 3: InstructionSelect crash on masked load of `<8 x i8>`

**File:** `bitcast_crash.ll`
**Error:** `LLVM ERROR: incompatible result and operand types in a bitcast`
**Location:** `InstructionSelect` pass

```bash
# Requires Bug 1 workaround (strip llvm.assume) to reach this bug.
llc -mtriple=spirv64-amd-amdhsa bitcast_crash.ll -o /dev/null
```

**Root cause:** The instruction selector internally generates a bitcast during
`llvm.masked.load` lowering that is incompatible in SPIR-V's type system.
The issue involves `<8 x i8>` vectors where SPIR-V cannot freely bitcast between
vector types with different element types/counts.

**Affected tests:** 2 IREE tests (dot i8xi8xi32, dot_bf16)

---

### Bug 4: Legalizer cannot handle `G_UNMERGE_VALUES` of `<64 x s32>`

**File:** `unmerge_crash.ll` (2385 lines — not reducible, tightly coupled shufflevector chain)
**Error:** `LLVM ERROR: unable to legalize instruction: G_UNMERGE_VALUES ... <64 x s32>`
**Location:** Legalization pass

```bash
# Requires Bug 1 workaround (strip llvm.assume) to reach this bug.
# The file already has llvm.assume stripped:
llc -mtriple=spirv64-amd-amdhsa unmerge_crash.ll -o /dev/null
```

**Root cause:** SPIR-V only supports vectors up to 16 elements. When LLVM IR builds
`<64 x float>` via shufflevector chains (winograd transform), the legalizer attempts
`G_UNMERGE_VALUES` to split `<64 x s32>` into `<16 x s32>` pieces but has no rule.
Running `opt -O1` on the file changes the crash to `G_BUILD_VECTOR` on `<64 x s32>`
(same underlying legalization gap, different entry point).

**Affected tests:** 1 IREE test (winograd_input)

---

## Unsupported Extensions (2 issues)

These are not crashes but produce SPIR-V that AMD HIP JIT rejects.

### Issue 5: Spurious `SPV_ALTERA_arbitrary_precision_integers` emission

**File:** `altera_extension.ll`
**Error (at runtime):** `Invalid SPIR-V module: input SPIR-V module uses unknown extension 'SPV_ALTERA_arbitrary_precision_integers'`

```bash
llc -mtriple=spirv64-amd-amdhsa -filetype=obj altera_extension.ll -o altera.spv
strings altera.spv | grep SPV_ALTERA
# Output: SPV_ALTERA_arbitrary_precision_integers
```

**Root cause:** The backend introduces non-standard integer widths during
strength reduction of `urem` by constant in loops, then declares the ALTERA
extension. The input IR only uses standard i32/i64 types. This extension is
not supported by AMD HIP SPIR-V JIT.

**Affected tests:** 6 IREE tests (broadcast, broadcast_add, broadcast_in_dim,
batch_norm_inference, householder, reduce_window)

---

### Issue 6: `SPV_INTEL_int4` for i4 integer types

**File:** `int4_extension.ll`
**Error (at runtime):** `Invalid SPIR-V module: input SPIR-V module uses unknown extension 'SPV_INTEL_int4'`

```bash
llc -mtriple=spirv64-amd-amdhsa -filetype=obj int4_extension.ll -o int4.spv
strings int4.spv | grep SPV_INTEL_int4
# Output: SPV_INTEL_int4
```

**Note:** This is arguably correct behavior from the SPIR-V backend — `i4` genuinely
requires an extension in SPIR-V. Fix options:
- IREE should widen i4 to i8 before SPIR-V emission
- AMD HIP SPIR-V JIT should support the i4 extension

**Affected tests:** 2 IREE tests (fp4_f32_conversion, small_float_arith)

---

## Test Summary

Out of 150 IREE HIP e2e tests:
- **116 pass** (77%)
- **26 "Not Run"** (compile-time crashes — Bugs 1-4)
- **8 "Failed"** (runtime failures — Issues 5-6)

After fixing Bug 1 (llvm.assume), the remaining compile-time crashes break down as:
- Bug 2 (vector GEP): 12 tests
- Bug 3 (masked load bitcast): 2 tests
- Bug 4 (G_UNMERGE large vectors): 1 test
- Other (winograd_output — unclear): 1 test
- Pre-existing non-SPIR-V issue: 1 test (narrow_n_matmuls — bufferize failure)

Tests that compile after Bug 1 fix but fail at runtime:
- Issue 5 (ALTERA extension): 6 tests
- Issue 6 (i4 extension): 2 tests
