// RUN: iree-opt --split-input-file --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-select-lowering-strategy, iree-llvmgpu-lower-executable-target)))" %s | FileCheck %s

// TODO: This test is still using the legacy LLVMGPU kernel config. This needs
// to be migrated to the rocdl heuristics, but for now is just physically
// located here.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @matmul_256x256x256_f16_f32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb", {
      target_arch = "gfx940",
      mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>,
                        #iree_gpu.mfma_layout<F16_32x32x8_F32>]
  }>) {
  hal.executable.export @matmul_256x256x256_f16_f32 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256_f16_f32() {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<256x256xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %5 = tensor.empty() : tensor<256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<256x256xf32>>
      return
    }
  }
}
}

// Basic pipeline test to make sure it generates the instructions we expect.

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute
//  CHECK-SAME:   mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>,
//  CHECK-SAME:     subgroup_m_count = 2, subgroup_n_count = 2, subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 2>

// CHECK-LABEL: hal.executable.export public @matmul_256x256x256_f16_f32
//  CHECK-SAME:    subgroup_size = 64
//  CHECK-SAME:    translation_info = #[[$TRANSLATION]]
//  CHECK-SAME:    workgroup_size = [128 : index, 2 : index, 1 : index]

//    CHECK-LABEL: func.func @matmul_256x256x256_f16_f32
//          CHECK:   scf.for {{.*}} = %c0 to %c256 step %c32 iter_args({{.*}}) -> (vector<2x4x1x1x1x4xf32>)
// Each subgroup handles 2 * 4 tiles, and for each tile we accumulate 2 times
// along the K dimension. So in total 16 mfma ops.
// CHECK-COUNT-16:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//          CHECK:     scf.yield %{{.+}} : vector<2x4x1x1x1x4xf32>
//  CHECK-COUNT-8:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<256x256xf32, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @matmul_256x256x256_f16_f16 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb", {
      target_arch = "gfx940",
      mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>,
                        #iree_gpu.mfma_layout<F16_32x32x8_F32>]
  }>) {
  hal.executable.export @matmul_256x256x256_f16_f16 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256_f16_f16() {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<256x256xf16>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %5 = tensor.empty() : tensor<256x256xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<256x256xf16>) -> tensor<256x256xf16>
      %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf16>) -> tensor<256x256xf16>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf16> -> !flow.dispatch.tensor<writeonly:tensor<256x256xf16>>
      return
    }
  }
}
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute
//  CHECK-SAME:   mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>,
//  CHECK-SAME:     subgroup_m_count = 2, subgroup_n_count = 2, subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 2>

// CHECK-LABEL: hal.executable.export public @matmul_256x256x256_f16_f16
//  CHECK-SAME:    subgroup_size = 64
//  CHECK-SAME:    translation_info = #[[$TRANSLATION]]
//  CHECK-SAME:    workgroup_size = [128 : index, 2 : index, 1 : index]

//    CHECK-LABEL: func.func @matmul_256x256x256_f16_f16
//          CHECK:   scf.for {{.*}} = %c0 to %c256 step %c32 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<2x4x1x1x1x4xf16>)
//          CHECK:     arith.extf %[[ARG]] : vector<2x4x1x1x1x4xf16> to vector<2x4x1x1x1x4xf32>
// CHECK-COUNT-16:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//          CHECK:     %[[TRUNC:.+]] = arith.truncf %{{.*}} : vector<2x4x1x1x1x4xf32> to vector<2x4x1x1x1x4xf16>
//          CHECK:     scf.yield %[[TRUNC]] : vector<2x4x1x1x1x4xf16>
//  CHECK-COUNT-8:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf16>, memref<256x256xf16, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @expanded_matmul_transpose_b_executable {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb", {
      target_arch = "gfx940",
      mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>,
                        #iree_gpu.mfma_layout<F16_32x32x8_F32>]
  }>) {
  hal.executable.export @expanded_matmul_transpose_b layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @expanded_matmul_transpose_b() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
          : !flow.dispatch.tensor<readonly:tensor<2x64x2048xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
          : !flow.dispatch.tensor<readonly:tensor<10x64x2048xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
          : !flow.dispatch.tensor<writeonly:tensor<2x10x64x64xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 64, 2048], strides = [1, 1, 1]
          : !flow.dispatch.tensor<readonly:tensor<2x64x2048xf16>> -> tensor<2x64x2048xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [10, 64, 2048], strides = [1, 1, 1]
          : !flow.dispatch.tensor<readonly:tensor<10x64x2048xf16>> -> tensor<10x64x2048xf16>

        %5 = tensor.empty() : tensor<2x10x64x64xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x10x64x64xf16>) -> tensor<2x10x64x64xf16>
        %7 = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
        } ins(%3, %4 : tensor<2x64x2048xf16>, tensor<10x64x2048xf16>) outs(%6 : tensor<2x10x64x64xf16>) {
        ^bb0(%lhs: f16, %rhs: f16, %out: f16):
          %mul = arith.mulf %lhs, %rhs : f16
          %add = arith.addf %mul, %out : f16
          linalg.yield %add : f16
        } -> tensor<2x10x64x64xf16>

        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 10, 64, 64], strides = [1, 1, 1, 1]
          : tensor<2x10x64x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x10x64x64xf16>>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable.export public @expanded_matmul_transpose_b
//  CHECK-SAME:    workgroup_size = [256 : index, 1 : index, 1 : index]

//    CHECK-LABEL: func @expanded_matmul_transpose_b
//          CHECK:   scf.for {{.*}} = %c0 to %c2048 step %c32 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<4x1x1x1x1x4xf16>)
//          CHECK:     arith.extf %[[ARG]] : vector<4x1x1x1x1x4xf16> to vector<4x1x1x1x1x4xf32>
// CHECK-COUNT-8:      amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//          CHECK:     %[[TRUNC:.+]] = arith.truncf %{{.*}} : vector<4x1x1x1x1x4xf32> to vector<4x1x1x1x1x4xf16>
//          CHECK:     scf.yield %[[TRUNC]] : vector<4x1x1x1x1x4xf16>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf16>, memref<2x10x64x64xf16, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @conv_nhwc_dispatch_0 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb", {
      target_arch = "gfx940",
      mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>,
                        #iree_gpu.mfma_layout<F16_32x32x8_F32>]
  }>) {
  hal.executable.export @conv_nhwc layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_nhwc() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x258x514x768xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x768x256xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x256x512x256xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 258, 514, 768], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x258x514x768xf16>> -> tensor<2x258x514x768xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 768, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x768x256xf16>> -> tensor<3x3x768x256xf16>
        %5 = tensor.empty() : tensor<2x256x512x256xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x258x514x768xf16>, tensor<3x3x768x256xf16>) outs(%6 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 256, 512, 256], strides = [1, 1, 1, 1] : tensor<2x256x512x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x256x512x256xf32>>
        return
      }
    }
  }
}

//    CHECK-LABEL: func.func @conv_nhwc
//          CHECK:   scf.for {{.*}} = %c0 to %c3
//          CHECK:     scf.for {{.*}} = %c0 to %c3
//          CHECK:       scf.for {{.*}} = %c0 to %c768 step %c32 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<2x4x1x1x1x4xf32>)
// CHECK-COUNT-16:         amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//  CHECK-COUNT-8:   vector.transfer_write {{.+}} : vector<4x1xf32>, memref<2x256x512x256xf32, #hal.descriptor_type<storage_buffer>>
