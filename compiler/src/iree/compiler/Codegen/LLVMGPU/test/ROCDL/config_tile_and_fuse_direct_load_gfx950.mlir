// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx950 \
// RUN:   --iree-llvmgpu-use-direct-load \
// RUN:   --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true \
// RUN:   --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN:   --iree-codegen-llvmgpu-use-igemm=false \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s \
// RUN:   | FileCheck %s

// Large f16 matmul — direct load should produce larger MN tiles than the
// non-direct-load path (subgroup N tile 8 vs 4, workgroup N tile 256 vs 128).
func.func @matmul_f16_direct_load(
    %arg0: tensor<4096x4096xf16>,
    %arg1: tensor<4096x4096xf16>,
    %arg2: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
  // CHECK-LABEL: func.func @matmul_f16_direct_load
  // CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
  // CHECK:   lowering_config = #iree_gpu.lowering_config
  // CHECK-SAME: mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>
  // CHECK-SAME: promote_operands = [0, 1]
  // CHECK-SAME: promotion_types = [#iree_gpu.use_global_load_dma, #iree_gpu.use_global_load_dma]
  // CHECK-SAME: subgroup = [4, 8, 0]
  // CHECK-SAME: workgroup = [128, 256, 0]
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<4096x4096xf16>, tensor<4096x4096xf16>)
                      outs(%arg2 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  return %0 : tensor<4096x4096xf32>
}
