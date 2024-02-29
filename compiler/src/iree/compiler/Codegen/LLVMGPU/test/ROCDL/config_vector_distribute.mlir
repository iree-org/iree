// RUN: iree-opt --split-input-file --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-select-lowering-strategy)))" %s | FileCheck %s

// TODO: This test is still using the legacy LLVMGPU kernel config. This needs
// to be migrated to the rocdl heuristics, but for now is just physically
// located here.

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[1, 1, 64, 64, 128]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>
// CHECK-SAME:   subgroup_m_count = 1, subgroup_n_count = 4,
// CHECK-SAME:   subgroup_m_tile_count = 4, subgroup_n_tile_count = 1, subgroup_k_tile_count = 8

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

// CHECK-LABEL: hal.executable public @expanded_matmul_transpose_b
// CHECK: linalg.generic {{.*}}lowering_config = #[[$TILE_SIZES]]

// -----

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[1, 1, 64, 128, 1, 1, 32]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>
// CHECK-SAME:   subgroup_m_count = 2, subgroup_n_count = 2,
// CHECK-SAME:   subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 2

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @conv_nhwc {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb", {
      target_arch = "gfx940",
      mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>,
                        #iree_gpu.mfma_layout<F16_32x32x8_F32>]
  }>) {
  hal.executable.export @conv_nhwc layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
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

// CHECK-LABEL: hal.executable public @conv_nhwc
// CHECK: linalg.conv_2d_nhwc_hwcf {{.*}} lowering_config = #[[$TILE_SIZES]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @matmul_256x256x256 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb", {
      target_arch = "gfx940",
      mma_intrinsics = []
  }>) {
  hal.executable.export @matmul_256x256x256 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256() {
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

// Check that we do not use the distribute pipeline if there are no supported
// intrinsics.
//       CHECK-NOT: iree_codegen.translation_info<LLVMGPUVectorDistribute

// -----

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[32, 128, 32]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>
// CHECK-SAME:   subgroup_m_count = 2, subgroup_n_count = 2,
// CHECK-SAME:   subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 4

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @matmul_1024x1024x1024 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb", {
      target_arch = "gfx940",
      mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>,
                        #iree_gpu.mfma_layout<F16_32x32x8_F32>]
  }>) {
  hal.executable.export @matmul_1024x1024x1024 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_1024x1024x1024() {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>> -> tensor<1024x1024xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>> -> tensor<1024x1024xf16>
      %5 = tensor.empty() : tensor<1024x1024xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%6 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: hal.executable public @matmul_1024x1024x1024
// CHECK: linalg.matmul {{.*}}lowering_config = #[[$TILE_SIZES]]
