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

**Impact:** The JIT compiler does NOT recover buffer instruction performance. Benchmarks
on gfx1201 (RDNA4) show **~20x slowdown** for compute-bound f32 matmul (2048³):
- AOT: 2.95 ms (5.82 TFLOPS) — 136 `buffer_load`, 0 scratch ops
- JIT: 57.8 ms (0.30 TFLOPS) — 0 `buffer_load`, 160 `global_load`, 1229 scratch spills

The JIT output lacks buffer instructions, dual-issue (`v_dual_*`), and suffers massive
register spilling. Bandwidth-bound kernels (elementwise add) are unaffected (1.0x).
See `bench/COMPARISON.md` for full analysis.

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

## LLVM SPIR-V Backend Bugs (9 bugs, 1 fixed upstream)

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

### Bug 5: SPIRVEmitIntrinsics crashes on aggregate `insertvalue` with PHI user

**File:** `aggregate_phi_crash.ll`
**Error:** `UNREACHABLE: illegal aggregate intrinsic user`
**Location:** `SPIRVEmitIntrinsics.cpp` (`replaceMemInstrUses`)

```bash
llc -mtriple=spirv64 aggregate_phi_crash.ll -o /dev/null
```

**Root cause:** `SPIRVEmitIntrinsics` rewrites `insertvalue` on aggregate types to
`@llvm.spv.insertv` and calls `replaceMemInstrUses()` to update all users of the
original instruction. That function handles `AssignType` intrinsics, memory
instructions, `ReturnInst`, and `CallInst` — but not `PHINode`. When the
`insertvalue` result feeds back into a loop PHI (common in reduction loops), the
unhandled-user UNREACHABLE fires.

**Affected tests:** 8 IREE tests (softmax, attention, dynamic_gather_attention,
linalg_ops_dynamic, split_reduction, dynamic_dot, dynamic_reduce_min, dot_general)

---

### Bug 6: `<1 x T>` nested in aggregates crashes `getOpTypeVector`

**File:** `vec1_crash.ll`
**Error:** `Assertion 'NumElems >= 2' failed` in `SPIRVGlobalRegistry::getOpTypeVector`
**Location:** `SPIRVGlobalRegistry.cpp:320`

```bash
llc -mtriple=spirv64-amd-amdhsa -filetype=obj vec1_crash.ll -o /dev/null
```

**Root cause:** SPIR-V's `OpTypeVector` requires at least 2 components. Upstream PR
[#180735](https://github.com/llvm/llvm-project/pull/180735) added scalarization of
`<1 x T>` → `T` in `getOrCreateSPIRVType()`, but the fix is **incomplete**: the
`createSPIRVType()` path reached through `findSPIRVType()` during recursive type
creation (e.g., processing array element types) has no `<1 x T>` guard. When
`<1 x T>` is nested inside an aggregate type like `[8 x <1 x float>]`, the crash
still triggers.

**Impact:** **Blocks ALL WMMA matmul types** — f16, bf16, i8, f8E4M3FN, f8E5M2.
IREE's WMMA codegen produces `[4 x [4 x [8 x <1 x float>]]]` types on gfx1201.

**Fix:** Add the same `<1 x T>` scalarization in `createSPIRVType()` (line ~1177).

**Affected tests:** All f16/bf16/i8 matmul e2e tests (8+ tests)

---

### Bug 7: `[1 x <N x T>]` array types cause PHI type mismatch in SPIR-V output

**File:** `array1_phi_mismatch.ll`
**Error (at JIT time):** `PHI node operands are not the same type as the result!`
**Location:** Mismatch between SPIR-V backend output and `amd-llvm-spirv -r` reverse translation

```bash
# Compile to SPIR-V (succeeds):
llc -mtriple=spirv64-amd-amdhsa -filetype=obj array1_phi_mismatch.ll -o out.spv
# Reverse-translate (fails):
amd-llvm-spirv -r out.spv -o out.bc
# Error: PHI node operands are not the same type as the result!
#   %N = phi <16 x float> [ %..., %... ], [ zeroinitializer, %entry ]
```

**Root cause:** When LLVM IR contains `phi [1 x <16 x float>]`, the SPIR-V backend
collapses the `[1 x T]` array to `T` for the PHI result type but keeps the array type
for `zeroinitializer` operand → produces inconsistent SPIR-V that `amd-llvm-spirv -r`
translates to type-mismatched LLVM IR. The kernel is then silently dropped by comgr.

**IREE workaround:** `unwrapSingleElementArrayTypes()` in `ROCMTarget.cpp` rewrites
`phi [1 x T]` → `phi T` and eliminates `insertvalue`/`extractvalue` wrappers before
SPIR-V codegen.

**Affected tests:** f32 matmul (non-WMMA). Without the IREE workaround, JIT produces
zeros or crashes.

---

### Bug 8: AMD vendor mode passes ALL intrinsics through as external functions

**File:** `intrinsic_passthrough.ll`
**Error (at JIT time):** Kernel silently dropped (unresolved `spirv.llvm_fma_v16f32`)
**Location:** `SPIRVPrepareFunctions.cpp:490`

```bash
# Produces SPIR-V with spirv.llvm_fma_v16f32 as external function declaration
# instead of lowering to OpenCL.std fma:
llc -mtriple=spirv64-amd-amdhsa -filetype=obj intrinsic_passthrough.ll -o out.spv
```

**Root cause:** The `default:` case in `SPIRVPrepareFunctions::lowerIntrinsicToFunction()`
converts ALL unhandled intrinsics to external function calls when
`Triple::Vendor == AMD`. This includes standard LLVM math intrinsics like `llvm.fma`,
`llvm.sin`, `llvm.cos`, etc. which the SPIR-V backend can lower natively via GlobalISel
to OpenCL.std extended instructions.

**Fix (applied locally):** Restrict the AMD pass-through to only AMDGCN-specific
intrinsics (`llvm.amdgcn.*`, `llvm.r600.*`). Standard math intrinsics are left for
the SPIR-V backend's normal lowering path.

**Affected tests:** Any kernel using FMA, trig, or other standard math intrinsics.

---

### Bug 9: `SPV_KHR_fma` extension emitted for AMD targets

**Error (at JIT time):** AMD HIP JIT rejects the `SPV_KHR_fma` extension
**Location:** `SPIRVSubtarget.cpp:102`

**Root cause:** When the SPIR-V backend enables all valid extensions for AMD targets,
`SPV_KHR_fma` is included. This extension maps `llvm.fma` to a dedicated SPIR-V
instruction. However, AMD's HIP SPIR-V JIT does not support `SPV_KHR_fma` — it expects
`fma` to go through `OpenCL.std` extended instruction set instead.

**Fix (applied locally):** Erase `SPV_KHR_fma` from AMD extension set so the backend
uses `OpenCL.std fma`.

**Affected tests:** Any kernel using `llvm.fma` intrinsic (e.g., matmul with FMA
accumulation).

---

## ROCm JIT Issues (comgr / amd-llvm-spirv)

These are bugs in the ROCm runtime's SPIR-V JIT pipeline, not in IREE or the LLVM
SPIR-V backend.

### JIT Issue 1: Silent kernel drop for some SPIR-V kernels

**Error:** No error — kernel executes but produces all-zero output
**Component:** ROCm comgr (Code Object Manager)

The comgr JIT pipeline (`amd-llvm-spirv -r` → LLVM opt → AMDGPU codegen → link)
silently drops the user kernel for some larger SPIR-V modules. The final code object
(`a.so`) contains only runtime builtins (`__amd_rocclr_*`), no user kernel.

**Observed for:** f32 matmul 2048³ and 4096³ transpose_b (46KB SPIR-V)
**NOT observed for:** f32 matmul 2048³ non-transposed (11KB SPIR-V), 1024³ transpose_b

The reverse-translated LLVM IR is valid and compiles fine with IREE's `llc` (LLVM 20).
The issue is specific to ROCm's LLVM 22 AMDGPU backend.

**Repro:**
```bash
# Compile with IREE:
iree-compile --iree-hal-target-device=hip --iree-rocm-target=gfx1201 \
  --iree-rocm-use-spirv matmul_tb.mlir -o out.vmfb

# Run with comgr debug:
AMD_COMGR_SAVE_TEMPS=1 iree-run-module --module=out.vmfb --device=hip \
  --function=... --input=2048x2048xf32 --input=2048x2048xf32

# Check: output is all zeros, and:
llvm-objdump -d /tmp/comgr-*/output/a.so | grep '<.*>:$'
# Shows only __amd_rocclr_* functions, no user kernel
```

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
- **116 pass** (77%) — zero JIT-side crashes
- **26 "Not Run"** (compile-time crashes — Bugs 1-5)
- **8 "Failed"** (runtime failures — Issues 6-7)

After fixing Bug 1 (llvm.assume, IREE workaround strips assumes), the remaining
compile-time crashes break down as:
- Bug 2 (vector GEP → now `isImm()` assertion after LLVM trunk updates): 10 tests
- Bug 3 (masked load bitcast): 2 tests
- Bug 4 (G_UNMERGE large vectors): 1 test
- Bug 5 (aggregate PHI): 8 tests
- SPIRVInstructionSelector UNREACHABLE: 1 test (select)
- IRTranslator crash: 1 test (winograd_output)
- Pre-existing non-SPIR-V issue: 1 test (narrow_n_matmuls — bufferize failure)

Tests that compile but fail at runtime:
- Issue 6 (ALTERA extension): 6 tests
- Issue 7 (i4 extension): 2 tests
