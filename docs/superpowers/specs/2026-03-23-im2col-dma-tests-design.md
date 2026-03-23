# Im2col DMA Test Design

## Goal

Add tests to validate the im2col DMA conversion pipeline for convolutions on
gfx950+. Two test types: an e2e test for numerical correctness on hardware, and
a pipeline lit test for IR-level verification in CI.

## Background

The im2col DMA path converts convolutions through:
`conv_2d_nhwc_hwcf` -> `im2col + matmul` (igemm) -> `im2col` gets
`UseGlobalLoadDMA` config -> `gather` -> `coalesced_gather_dma` ->
`amdgpu.gather_to_lds`.

This path is enabled by `--iree-llvmgpu-use-direct-load=true` combined with
`--iree-codegen-llvmgpu-use-igemm=true` and requires gfx950+ target.

### DMA Alignment Constraint

On gfx950 with `dma_sizes=[32, 128]` bits and `subgroup_size=64`, each thread
must handle at least 32 bits (4 bytes) of contiguous data. For f16 (2 bytes per
element), the tiled K dimension of the im2col output must satisfy:

    tiled_K * 2 / 64 >= 4 bytes  =>  tiled_K >= 128

With MFMA_F32_16x16x32_F16 (Kb=32), this requires a reduction tile of at least
4 MMA blocks. Channels must be large enough for the config selector to produce
such tiles.

## Test 1: E2E Test

**File**: `tests/e2e/rocm_specific/im2col_dma_conv.mlir`

**Convolution**:
- Input: `1x10x10x512xf16`, all ones
- Filter: `3x3x512x512xf16`, all ones
- Output: `1x8x8x512xf32`
- Stride 1, dilation 1, no padding
- Expected result: every element = 3 * 3 * 512 = 4608.0

**Structure** (follows `lds_matmul.mlir` pattern):
```mlir
func.func @im2col_dma_conv() {
  %input = util.unfoldable_constant dense<1.0> : tensor<1x10x10x512xf16>
  %filter = util.unfoldable_constant dense<1.0> : tensor<3x3x512x512xf16>
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x8x8x512xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : ...) -> ...
  %result = linalg.conv_2d_nhwc_hwcf {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : ...) outs(%fill : ...) -> tensor<1x8x8x512xf32>
  check.expect_almost_eq_const(
    %result, dense<4608.0> : tensor<1x8x8x512xf32>)
    : tensor<1x8x8x512xf32>
  return
}
```

**Compilation** (user runs on gfx950 machine):
```bash
iree-compile \
  --iree-hal-target-backends=rocm \
  --iree-rocm-target=gfx950 \
  --iree-codegen-llvmgpu-use-igemm=true \
  --iree-llvmgpu-use-direct-load=true \
  im2col_dma_conv.mlir -o im2col_dma_conv.vmfb
```

**Execution**:
```bash
iree-check-module --device=hip --module=im2col_dma_conv.vmfb
```

**IR dump** (for debugging):
```bash
iree-compile \
  --iree-hal-target-backends=rocm \
  --iree-rocm-target=gfx950 \
  --iree-codegen-llvmgpu-use-igemm=true \
  --iree-llvmgpu-use-direct-load=true \
  --mlir-print-ir-after-all \
  im2col_dma_conv.mlir -o im2col_dma_conv.vmfb 2> im2col_dma_ir_dump.mlir
```

**Test registration**: Manual-only. `tests/e2e/rocm_specific/CMakeLists.txt` is
currently empty with no build wiring. No CMakeLists/BUILD.bazel changes needed.

## Test 2: Pipeline Lit Test

**File**: `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_im2col_dma_gfx950.mlir`

The test uses the same conv shape as the e2e test, wrapped in
`hal.executable` / `hal.executable.variant` boilerplate targeting gfx950.

### Structure

The test has a single RUN line running the full pipeline with a manually frozen
lowering config:

```
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(
// RUN:     builtin.module(func.func(iree-llvmgpu-lower-executable-target
// RUN:       {for-rocdl=true})))))" %s | FileCheck %s
```

### Frozen Config

The `#translation` and `#config` attributes are obtained by running strategy
selection on the conv with `--iree-llvmgpu-use-direct-load=true` and
`--iree-codegen-llvmgpu-use-igemm=true` on gfx950, then freezing the output.
The config must include:
- `promotion_types` with `#iree_gpu.use_global_load_dma` for operands 0 and 1
- `use_igemm_convolution = true` in `gpu_pipeline_options`
- `no_reduce_shared_memory_bank_conflicts = true` (set by direct load path)
- Appropriate `workgroup`, `reduction`, `subgroup` tiles and `mma_kind`

During implementation, the frozen config will be captured by running:
```bash
iree-opt --iree-gpu-test-target=gfx950 \
  --iree-codegen-llvmgpu-use-igemm=true \
  --iree-llvmgpu-use-direct-load=true \
  --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" \
  <input.mlir>
```

### CHECK Patterns

Verify the lowered IR contains `amdgpu.gather_to_lds` for the im2col input
operand loading into LDS. Check for source-shape-specific gather patterns
rather than bare `amdgpu.gather_to_lds` presence.

### Build Registration

The new lit test file must be added to:
- `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/CMakeLists.txt`
- `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/BUILD.bazel`

Both use explicit file lists.

## Files to Create/Modify

1. **Create** `tests/e2e/rocm_specific/im2col_dma_conv.mlir` - e2e test
2. **Create** `compiler/.../LLVMGPU/test/ROCDL/pipeline_im2col_dma_gfx950.mlir`
3. **Modify** `compiler/.../LLVMGPU/test/ROCDL/CMakeLists.txt` - add new test
4. **Modify** `compiler/.../LLVMGPU/test/ROCDL/BUILD.bazel` - add new test

## Shape Rationale

`1x10x10x512xf16` input with `3x3x512x512xf16` filter was chosen because:
- C=512 satisfies the DMA alignment check on the im2col output's contiguous
  slice. The `isIm2colDMAConvertible` check gates on the vectorizable
  contiguous dimension (C, not the full 9*C). With subgroup_size=64 and
  dma_sizes=[32,128] bits, the minimum contiguous f16 dimension is 128
  elements (512 >= 128, and 512 % 128 == 0).
- Spatial 8x8 output gives M = 64, sufficient for MFMA tile sizes
- N = 512 output channels is standard for MMA scheduling
- Expected value 4608.0 is trivially computable (3 * 3 * 512)
- C=64 was rejected: with subgroup_size=64 and f16, per-thread data is
  only 16 bits, below the 32-bit minimum DMA transfer size
