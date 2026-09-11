# LLVMCPU C microkernels

This directory holds **C-based microkernels** (ukernels) for the LLVMCPU
target backend. They are compiled to LLVM bitcode, embedded as static data
in `iree-compile`, and at compile time are copied into the IR as
`hal.executable_object` attributes on the dispatch's HAL executable variant.
That representation is the same as the GPU C ukernels under
`compiler/plugins/target/ROCM/builtins/ukernel/`.

This is **not** the same framework as the legacy CPU ukernels under
`runtime/src/iree/builtins/ukernel/`. The two coexist; this directory is
purely additive. The table below summarizes the differences at a glance; the
[design pillars](#design-pillars) section expands on the most consequential
ones.

## Relationship to the legacy CPU ukernels

| Aspect | Legacy (`runtime/src/iree/builtins/ukernel/`) | New (this directory) |
|---|---|---|
| Location in the tree | `runtime/src/iree/builtins/ukernel/` | `compiler/plugins/target/LLVMCPU/builtins/ukernel/` |
| Expected prevalence | The *majority* path for data-tiled matmul on CPU, back when codegen generally could not lower `mmt4d` to hardware intrinsics. | A *minority* path for the new `inner_tiled` data-tiled matmul: codegen *can* now lower `inner_tiled` to intrinsics, so a ukernel is the exception, not the rule. |
| Consumers | LLVMCPU, VMVX, and standalone tests. | LLVMCPU only — the VMVX consumer that justified a runtime-side location is defunct. |
| Build | Compiled *twice*: as LLVM bitcode and as native code (the latter for VMVX and standalone tests). | Compiled *once*, as LLVM bitcode only. |
| MLIR representation | An opaque external reference the runtime is expected to provide. | Pulled into a `hal.executable_object` carried in the IR, making the module self-contained. |
| Bring-your-own | Fork the IREE source tree. | Attach your own `hal.executable_object` to your input MLIR (see [below](#how-to-bring-your-own-ukernel)). |
| Call overhead | The dual build and multiple consumers left it unclear whether dispatch overhead would need amortizing; by the time it was clear that LLVMCPU inlines ukernels into their callers, the design was already fixed. | Designed from scratch around being always bitcode, always inlined, and specialized into the caller: seamless, with no residual call overhead. |
| Op granularity | Coarser — the `mmt4d` ukernel owns the outer M/N loops itself. | Finer — the `inner_tiled` ukernel owns only the innermost K loop. |
| Genericity | The native build forces every variant (e.g. narrow-shape matmuls) to be instantiated in the ukernel sources. | The bitcode-only build plus late inlining lets a ukernel take extra parameters that callers pass as compile-time constants, specializing the body — so e.g. narrow shapes largely become simple truncations of the general tile rather than separate ukernels. |

## Scope

The framework itself is **not tied to any one op**. A ukernel is just a C
function, compiled to bitcode, that codegen may substitute for some piece
of IR it would otherwise lower itself. Any op that codegen knows how to
match and replace with a `iree_codegen.ukernel.generic` call can have a
ukernel; the build/embed/lookup/lowering machinery below is generic.

The **initial and primary** use case — and the only one wired up so far —
is `iree_codegen.inner_tiled`. The rest of this document leans on that case
for its concrete examples, but keep the distinction in mind: the pillars
about *where ukernels live* and *how their bitcode reaches the IR* are
general, while the specifics of *what a ukernel computes* (the inner K
loop, the `intrinsics_{m,n,k}` contract) are particular to `inner_tiled`.

### The `inner_tiled` case

For an `iree_codegen.inner_tiled` op with a
`#iree_cpu.data_tiled_mma_layout`, a ukernel implements the **inner K
loop** of one data-tiled MMA tile. Outer M/N looping is handled by ordinary
IREE tiling *before* the ukernel runs; the ukernel sees a single
(M, N) tile and walks K itself.

A ukernel is specific to **one `MMAIntrinsic`** but **generic over the
`intrinsics_{m,n,k}` unrolling factors**. Those factors are passed to the C
function as ordinary parameters and look like runtime values inside the
ukernel's translation unit — but they are *always* compile-time constants
in the calling context (the caller passes the `DataTiledMMAAttr`'s
constants), and the ukernel is *always* inlined into that caller (anything
else is a bug). Linking the ukernel as bitcode into the dispatch lets the
post-inlining optimizer fully specialize the body to each call site's
concrete `(intrinsics_m, intrinsics_n, intrinsics_k)` triple: loops over
those counts unroll, the unrolled tile lives in registers, and nothing of
the "runtime parameter" framing survives. In effect the bitcode + inlining
+ LTO chain gives C++-template-like specialization without the template
syntax — specialization on the `MMAIntrinsic` happens via *symbol
selection* (one ukernel function per intrinsic), and specialization on the
unrolling factors happens via *post-inlining loop unrolling*.

Zooming in on the *interface contract* (the
[comparison above](#relationship-to-the-legacy-cpu-ukernels) is at the
framework level), this is a **lower-level interface than the legacy
`mmt4d` ukernels**:

| | Legacy (`runtime/.../ukernel/`) | New (this directory) |
|---|---|---|
| What an entry point is | A whole-matmul library: handles arbitrary `mmt4d` shapes, walks outer M/N itself, internally dispatches to arch-variant inner kernels | The inner K-loop of one specific (intrinsic, arch-variant) configuration |
| Outer M/N looping | Inside the ukernel | Outside, done by IREE tiling |
| Per-intrinsic specialization | Hidden behind a dispatching front door | Exposed — one entry point per (intrinsic, arch-variant) |
| Caller responsibility | Almost nothing | Tile to the ukernel's exact data-tiled shape |

## Design pillars

### 1. Lives in `compiler/`, built only as LLVM bitcode

Two structural decisions, both intentionally different from the legacy
framework:

- **`compiler/`, not `runtime/`.** The legacy ukernels live under
  `runtime/src/iree/builtins/ukernel/` because at the time they served two
  consumers: the llvm-cpu backend *and* VMVX. The VMVX ukernel path is now
  effectively defunct, so the runtime-side / shared-with-VMVX
  justification no longer applies. The new ukernels are llvm-cpu-only and
  live in the plugin that uses them, mirroring the GPU C ukernels under
  `compiler/plugins/target/ROCM/builtins/ukernel/`.

- **Bitcode only.** The legacy ukernels are also built with the native
  toolchain, to be testable/benchmarkable in isolation via the runtime's
  `tools/` directory. That property turned out not to be worth the extra
  build complexity. The new ukernels are built only as bitcode, embedded
  as static data in `iree-compile`, and tested via the self-contained-IR
  property described next.

### 2. IR representation: self-contained via `hal.executable_object`

When `iree-compile` decides to use one of these built-in ukernels, the
ukernel's bitcode is **copied into the IR** as a `hal.executable_object`
attached to the dispatch's executable variant. From that point on the
module is self-contained: everything needed to compile and run the
dispatch is in the IR.

This is **different** from the legacy framework, where the ukernel was an
opaque external symbol the runtime was expected to provide. Two
consequences fall out:

- **Tests are trivially writable.** A lit test is just an MLIR file that
  carries its own `hal.executable_object`. The test owns its ukernel
  bitecode bytes outright — no runtime-side fixturing, no special CI
  wiring, no need to actually rerun the bitcode-library build for the test
  to run.
- **Bring-your-own-ukernel is just `hal.executable_object`.** A user who
  wants to override or supply their own ukernel attaches their own
  `hal.executable_object` to their input MLIR. `iree-compile` honors it on
  equal footing with the built-in ones (and the lookup logic, when it
  walks up looking for executable objects, finds the user's one first).
  No fork of IREE, no runtime patching.

## How a built-in ukernel works end to end

This walks through the `inner_tiled` path; an analogous flow would apply to
any future op the framework grows to cover.

1. **Source.** A C file under this directory implements the ukernel as a
   single `__attribute__((always_inline))` function. For `inner_tiled`, the
   function takes tile pointers + the `intrinsics_{m,n,k}` unrolling factors
   as scalar arguments, and emits the unrolled inner K-loop. The file does
   `#include "common.h"` for no-stdlib stdint replacements and pulls in
   architecture-specific intrinsic headers (e.g. `<immintrin.h>` for x86).

2. **Build.** A bazel/CMake rule (`iree_bitcode_library` with `ARCH=x86_64`
   and feature-specific `COPTS` like `-mavx512bf16`) compiles the C file
   to an LLVM bitcode file, e.g.
   `iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16.x86_64_avx512bf16.bc`.

3. **Embed.** `iree_c_embed_data` collects all such `.bc` files into a
   single TOC blob (`iree_uk_cpu_bitcode.{c,h}`), which is linked into
   `iree-compile`.

4. **Register at compile time.** On LLVMCPU target init, the TOC is
   iterated and the bitcode files are inserted into the global
   `EmbeddedDataDirectory`.

5. **Match.** During LLVMCPU's kernel configuration step
   (`LLVMCPUSelectUKernels`), an `iree_codegen.inner_tiled` op carrying a
   `#iree_cpu.data_tiled_mma_layout` is matched against the available
   ukernels by intrinsic + element types + arch + features. If a match is
   found, the op gets a `iree_codegen.ukernel = "<name>"` descriptor
   attribute set on it, and the matching bitcode is attached.

6. **Lower.** The generic pass
   `iree-codegen-lower-bitcode-ukernels` finds annotated ops, looks up the
   `iree_codegen.ukernel_provider` (a
   `#iree_cpu.ukernel_provider` attribute installed on the target config),
   and calls its `createAndReplaceWithUkernelOp`. That method:
   - Looks up the matching `hal.executable_object` (first in the
     dispatch's existing `hal.executable.objects`, then in the global
     `EmbeddedDataDirectory`).
   - Attaches it to the executable variant.
   - Replaces the op with an `iree_codegen.ukernel.generic` call to the
     ukernel, with `fn_def_attrs = {hal.import.bitcode = true}` so the
     eventual call resolves directly against the linked bitcode rather
     than through the runtime import table.

7. **Lower again.** The existing
   `iree-codegen-lower-ukernel-ops-to-calls` pass lowers
   `ukernel.generic` to a `func.call` against the (now-linked) bitcode
   function. The LLVM `always_inline` attribute then specializes the
   ukernel body into each call site at the LLVM optimization stage.

## How to use as an IREE end user

Pass `--iree-llvmcpu-enable-llvm-ukernels=inner_tiled` to `iree-compile`.
The flag takes a comma-separated list of categories to enable (currently
just `inner_tiled`; the name leaves room for future categories). When a
category is enabled, any matching op implemented by one of the built-in
ukernels listed below will be rewritten to a ukernel call, and the bitcode
will appear in the resulting MLIR as a `hal.executable_object` on the
executable variant.

## How to bring your own ukernel

Attach a `hal.executable_object` to the input MLIR carrying your bitcode
(with a function whose name matches a known ukernel name). When
`iree-compile` walks up looking for executable objects, it finds your one
first and uses it in place of the built-in. Built-in ukernels are
overridden seamlessly; new ukernels (functions IREE has no built-in for)
work the same way as long as the user-supplied IR also annotates the
matched op with `iree_codegen.ukernel = "<name>"`.

## How to author a new built-in ukernel

1. Write the C source under this directory. Function name (and file name)
   is the corresponding `MMAIntrinsic` enum value, lowercased, with the
   `iree_uk_` prefix — e.g. `MMA_X86_AVX512BF16_1x16x2_F32_BF16` becomes
   `iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16`. This mirrors the AMDGPU C
   ukernels: those carry the AMDGPU intrinsic name (`mfma_i32_16x16x32_i8`,
   etc.) verbatim, and on CPU the arch tag (`x86`, ...) is already part of
   the intrinsic name, so this convention drops in cleanly. One ukernel per
   intrinsic — the `intrinsics_{m,n,k}` unrolling factors are *function
   arguments*, not part of the name (see Scope above for why this fully
   specializes anyway). Body is `always_inline`, no allocations, no stdlib
   (just `common.h` and intrinsic headers). Loops driven by the unrolling
   factors should unroll spontaneously after inlining; if Clang doesn't
   manage on its own, add a targeted `#pragma clang loop unroll(...)`.
2. Add a `iree_bitcode_library` entry in `BUILD.bazel` / `CMakeLists.txt`
   for the new file, with the appropriate `COPTS` feature flags, and add
   the resulting `.bc` to the embedded TOC.
3. If the ukernel implements an `MMAIntrinsic` that the codegen pipeline
   does not lower, add a new enum value to `IREECPUEnums.td` and minimal
   metadata (`getRowMajorTilesMNKShape`, `getABCElementTypes`) in
   `IREECPUAttrs.cpp` — *but no `lowerX86…` function*, because the
   ukernel is the sole implementation by design.
4. Extend `LLVMCPUSelectUKernels.cpp` to recognize the new ukernel by
   intrinsic + element types + arch features.

## How to test

Two complementary levels:

- **Lit tests (compiler IR).** Use the self-contained-IR property: write a
  lit test that carries the ukernel bitcode directly as a
  `hal.executable_object` literal in the MLIR. `iree-opt` running the
  lowering passes can verify the resulting IR without ever invoking the
  bitcode-library build. See the lit tests under `test/` for examples —
  ranging from single-pass checks (`lower_inner_tiled_to_bitcode_ukernel*`,
  `select_ukernel`) up to a full-codegen-pipeline check
  (`e2e_inner_tiled_pipeline`) that drives an `inner_tiled` dispatch all the
  way to a direct `llvm.call` against the ukernel symbol.

- **End-to-end numerical tests (`tests/e2e/matmul`).** For actually running
  a ukernel and checking its results against a reference, add a variant to
  the matmul e2e runner tests under `tests/e2e/matmul/` (the
  `iree_tests_e2e_matmul_*` family), compiling with
  `--iree-llvmcpu-enable-llvm-ukernels=inner_tiled`. This is the level that
  catches a wrong inner loop — a lit test only checks that the *call* is
  emitted, not that the ukernel *computes* the right thing.

## Seed examples

The directory is intentionally **sparse**. Ukernels are no longer the
majority path under the modernized CPU codegen; codegen handles most
cases on its own. The seeds here illustrate two categories that justify
a ukernel at all:

| Seed | What it shows |
|---|---|
| `iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16.c` | Clean example. Codegen does the same thing at parity; this is the canonical "simplest new-style ukernel" reference for new authors. |
| `iree_uk_mma_x86_avx512vnni_16x16x2_i32_i8_casti16.c` | Practically useful. i8×i8→i32 via VNNI is a workhorse for quantized inference, and codegen has a residual perf gap on this case. |

This README is kept up to date as new seeds and framework pieces land.
