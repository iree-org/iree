// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s

func.func @multi_mma_mfma_i32_16x16x32_i8(%a : tensor<1x2x8x4x16x2x8xi8>,
                                %b : tensor<1x2x4x2x4x16x2x8xi8>,
                                %c : tensor<1x1x8x4x2x4x16x4xi32>)
    -> tensor<1x1x8x4x2x4x16x4xi32> attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {ukernels = "multi_mma"}>
} {
  %d = iree_codegen.inner_tiled ins(%a, %b) outs(%c) {indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ], iterator_types = [
      #linalg.iterator_type<parallel>,
      #linalg.iterator_type<parallel>,
      #linalg.iterator_type<reduction>
    ], kind = #iree_gpu.data_tiled_mma_layout<
      intrinsic =  MFMA_I32_16x16x32_I8,
      intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2
    >} : tensor<1x2x8x4x16x2x8xi8>, tensor<1x2x4x2x4x16x2x8xi8> into tensor<1x1x8x4x2x4x16x4xi32>
  return %d : tensor<1x1x8x4x2x4x16x4xi32>
}

// CHECK-LABEL: @multi_mma_mfma_i32_16x16x32_i8
//       CHECK: iree_codegen.inner_tiled
//  CHECK-SAME: #hal.executable.object<{path = "iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8.gfx942.bc"
//  CHECK-NOT:  promote_operands
//  CHECK-SAME: reduction = [0, 0, 0]
//  CHECK-SAME: #iree_gpu.ukernel_config<name = "iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8"
//  CHECK-SAME: shared_memory_bytes = 8192
