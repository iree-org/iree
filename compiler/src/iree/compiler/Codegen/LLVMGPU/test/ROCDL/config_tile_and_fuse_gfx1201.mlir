// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx1201 \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s \
// RUN:   | FileCheck %s --check-prefix=GEMM
// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx1201 \
// RUN:   --iree-codegen-llvmgpu-use-igemm=false --iree-codegen-llvmgpu-use-direct-convolution=true \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s \
// RUN:   | FileCheck %s --check-prefix=CONV

// Verify RDNA4-specific heuristic seed selection produces expected configs for
// matmul and convolution operations at different arithmetic intensity levels.

// ============================================================================
// Matmul — small (low arithmetic intensity, memory-bound)
// ============================================================================

func.func @matmul_small_f16(%arg0: tensor<64x1280xf16>, %arg1: tensor<1280x1280xf16>) -> tensor<64x1280xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<64x1280xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x1280xf32>) -> tensor<64x1280xf32>
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<64x1280xf16>, tensor<1280x1280xf16>) outs(%fill : tensor<64x1280xf32>) -> tensor<64x1280xf32>
  return %result : tensor<64x1280xf32>
}

// GEMM-LABEL: func.func @matmul_small_f16
//  GEMM-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
//  GEMM-SAME:   workgroup_size = [128, 1, 1] subgroup_size = 32
//       GEMM:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  GEMM-SAME:     mma_kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>
//  GEMM-SAME:     promote_operands = [0, 1]
//  GEMM-SAME:     reduction = [0, 0, 4]
//  GEMM-SAME:     subgroup = [2, 2, 0]
//  GEMM-SAME:     workgroup = [64, 64, 0]

// -----

// ============================================================================
// Matmul — medium (moderate arithmetic intensity)
// ============================================================================

func.func @matmul_medium_f16(%arg0: tensor<2048x1280xf16>, %arg1: tensor<1280x1280xf16>) -> tensor<2048x1280xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<2048x1280xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%fill : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>
  return %result : tensor<2048x1280xf32>
}

// GEMM-LABEL: func.func @matmul_medium_f16
//  GEMM-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
//  GEMM-SAME:   workgroup_size = [128, 1, 1] subgroup_size = 32
//       GEMM:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  GEMM-SAME:     mma_kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>
//  GEMM-SAME:     promote_operands = [0, 1]
//  GEMM-SAME:     reduction = [0, 0, 4]
//  GEMM-SAME:     subgroup = [2, 2, 0]
//  GEMM-SAME:     workgroup = [64, 64, 0]

// -----

// ============================================================================
// Matmul — large (high arithmetic intensity, compute-bound)
// ============================================================================

func.func @matmul_large_f16(%arg0: tensor<4096x4096xf16>, %arg1: tensor<4096x4096xf16>) -> tensor<4096x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4096x4096xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<4096x4096xf16>, tensor<4096x4096xf16>) outs(%fill : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  return %result : tensor<4096x4096xf32>
}

// GEMM-LABEL: func.func @matmul_large_f16
//  GEMM-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
//  GEMM-SAME:   workgroup_size = [256, 1, 1] subgroup_size = 32
//       GEMM:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  GEMM-SAME:     mma_kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>
//  GEMM-SAME:     promote_operands = [0, 1]
//  GEMM-SAME:     reduction = [0, 0, 4]
//  GEMM-SAME:     subgroup = [4, 4, 0]
//  GEMM-SAME:     workgroup = [256, 128, 0]

// -----

// ============================================================================
// Convolution — small (low arithmetic intensity)
// ============================================================================

func.func @conv_small_f16(%arg0: tensor<1x18x18x16xf16>, %arg1: tensor<32x3x3x16xf16>) -> tensor<1x16x16x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<1x16x16x32xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x16x16x32xf32>) -> tensor<1x16x16x32xf32>
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%arg0, %arg1 : tensor<1x18x18x16xf16>, tensor<32x3x3x16xf16>)
      outs(%fill : tensor<1x16x16x32xf32>) -> tensor<1x16x16x32xf32>
  return %0 : tensor<1x16x16x32xf32>
}

// CONV-LABEL: func.func @conv_small_f16
//  CONV-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
//  CONV-SAME:   workgroup_size = [128, 1, 1] subgroup_size = 32
//       CONV:   linalg.conv_2d_nhwc_fhwc {{.*}}lowering_config = #iree_gpu.lowering_config
//  CONV-SAME:     mma_kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>
//  CONV-SAME:     promote_operands = [0, 1]
//  CONV-SAME:     reduction = [0, 0, 0, 0, 1, 1, 1]
//  CONV-SAME:     subgroup = [1, 2, 1, 1, 0, 0, 0]
//  CONV-SAME:     workgroup = [1, 4, 16, 32, 0, 0, 0]

// -----

// ============================================================================
// Convolution — medium (moderate arithmetic intensity)
// ============================================================================

func.func @conv_medium_f16(%arg0: tensor<2x66x66x128xf16>, %arg1: tensor<128x3x3x128xf16>) -> tensor<2x64x64x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<2x64x64x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<2x64x64x128xf32>) -> tensor<2x64x64x128xf32>
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%arg0, %arg1 : tensor<2x66x66x128xf16>, tensor<128x3x3x128xf16>)
      outs(%fill : tensor<2x64x64x128xf32>) -> tensor<2x64x64x128xf32>
  return %0 : tensor<2x64x64x128xf32>
}

// CONV-LABEL: func.func @conv_medium_f16
//  CONV-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
//  CONV-SAME:   workgroup_size = [128, 1, 1] subgroup_size = 32
//       CONV:   linalg.conv_2d_nhwc_fhwc {{.*}}lowering_config = #iree_gpu.lowering_config
//  CONV-SAME:     mma_kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>
//  CONV-SAME:     promote_operands = [0, 1]
//  CONV-SAME:     reduction = [0, 0, 0, 0, 1, 1, 4]
//  CONV-SAME:     subgroup = [1, 1, 2, 2, 0, 0, 0]
//  CONV-SAME:     workgroup = [1, 1, 64, 64, 0, 0, 0]

// -----

// ============================================================================
// Convolution — large (high arithmetic intensity, compute-bound)
// ============================================================================

func.func @conv_large_f16(%arg0: tensor<16x50x34x576xf16>, %arg1: tensor<576x3x3x576xf16>) -> tensor<16x48x32x576xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<16x48x32x576xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<16x48x32x576xf32>) -> tensor<16x48x32x576xf32>
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%arg0, %arg1 : tensor<16x50x34x576xf16>, tensor<576x3x3x576xf16>)
      outs(%fill : tensor<16x48x32x576xf32>) -> tensor<16x48x32x576xf32>
  return %0 : tensor<16x48x32x576xf32>
}

// CONV-LABEL: func.func @conv_large_f16
//  CONV-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
//  CONV-SAME:   workgroup_size = [128, 1, 1] subgroup_size = 32
//       CONV:   linalg.conv_2d_nhwc_fhwc {{.*}}lowering_config = #iree_gpu.lowering_config
//  CONV-SAME:     mma_kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>
//  CONV-SAME:     promote_operands = [0, 1]
//  CONV-SAME:     reduction = [0, 0, 0, 0, 1, 1, 4]
//  CONV-SAME:     subgroup = [1, 2, 1, 2, 0, 0, 0]
//  CONV-SAME:     workgroup = [1, 2, 32, 64, 0, 0, 0]
