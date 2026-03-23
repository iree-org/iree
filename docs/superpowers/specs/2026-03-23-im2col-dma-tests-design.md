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

## Test 1: E2E Test

**File**: `tests/e2e/rocm_specific/im2col_dma_conv.mlir`

**Convolution**:
- Input: `1x8x8x64xf16`, all ones
- Filter: `3x3x64x128xf16`, all ones
- Output: `1x6x6x128xf32`
- Stride 1, dilation 1, no padding
- Expected result: every element = 3 * 3 * 64 = 576.0

**Structure** (follows `lds_matmul.mlir` pattern):
```mlir
func.func @im2col_dma_conv() {
  %input = util.unfoldable_constant dense<1.0> : tensor<1x8x8x64xf16>
  %filter = util.unfoldable_constant dense<1.0> : tensor<3x3x64x128xf16>
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x6x6x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : ...) -> ...
  %result = linalg.conv_2d_nhwc_hwcf {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : ...) outs(%fill : ...) -> tensor<1x6x6x128xf32>
  check.expect_almost_eq_const(%result, dense<576.0> : tensor<1x6x6x128xf32>)
    : tensor<1x6x6x128xf32>
  return
}
```

**Compilation command** (user runs on gfx950 machine):
```bash
iree-compile \
  --iree-hal-target-backends=rocm \
  --iree-hip-target=gfx950 \
  --iree-codegen-llvmgpu-use-igemm=true \
  --iree-llvmgpu-use-direct-load=true \
  im2col_dma_conv.mlir -o im2col_dma_conv.vmfb
```

**IR dump command** (for debugging):
```bash
iree-compile \
  --iree-hal-target-backends=rocm \
  --iree-hip-target=gfx950 \
  --iree-codegen-llvmgpu-use-igemm=true \
  --iree-llvmgpu-use-direct-load=true \
  --mlir-print-ir-after-all \
  im2col_dma_conv.mlir -o im2col_dma_conv.vmfb 2> im2col_dma_ir_dump.mlir
```

## Test 2: Pipeline Lit Test

**File**: `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_im2col_dma_gfx950.mlir`

**RUN line 1 - Strategy selection**:
```
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 \
// RUN:   --iree-codegen-llvmgpu-use-igemm=true \
// RUN:   --iree-llvmgpu-use-direct-load=true \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" \
// RUN:   %s | FileCheck %s --check-prefix=CHECK-CONFIG
```

Verifies `use_global_load_dma` appears in the lowering config for im2col
operand promotion.

**RUN line 2 - Full pipeline lowering**:
```
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(
// RUN:     builtin.module(func.func(iree-llvmgpu-lower-executable-target
// RUN:       {for-rocdl=true})))))" %s | FileCheck %s --check-prefix=CHECK-LOWER
```

Uses manually specified lowering config with `use_global_load_dma`. Verifies
the lowered IR contains `amdgpu.gather_to_lds` for the im2col input operand.

**Test case**: Same conv shape as the e2e test (1x8x8x64 input, 3x3x64x128
filter, f16, stride 1). Wrapped in `hal.executable` / `hal.executable.variant`
boilerplate targeting gfx950.

**CHECK patterns**:
- `CHECK-CONFIG`: `use_global_load_dma` in promotion_types
- `CHECK-LOWER`: `amdgpu.gather_to_lds` present, no `vector.transfer_read`
  from `fat_raw_buffer` for the im2col operand

## Files to Create

1. `tests/e2e/rocm_specific/im2col_dma_conv.mlir` - e2e test
2. `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_im2col_dma_gfx950.mlir` - pipeline lit test
3. Update CMakeLists if needed for the lit test registration

## Shape Rationale

The small shape (1x8x8x64) was chosen because:
- Channel dim 64 with f16 = 128 bytes, well-aligned for DMA transfer sizes
- Output spatial 6x6 is small enough that expected values are trivially computed
- Exercises the full im2col DMA path without excessive compile/run time
