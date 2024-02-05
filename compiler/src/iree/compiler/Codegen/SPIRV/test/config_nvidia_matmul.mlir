// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-select-lowering-strategy-pass)))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @matmul_4x4096x9216 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Shader], []>, NVIDIA:DiscreteGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 32>>
    }>) {
    hal.executable.export @matmul_4x4096x9216 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_4x4096x9216() {
        %c36864 = arith.constant 36864 : index
        %c667974912 = arith.constant 667974912 : index
        %c209920 = arith.constant 209920 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x9216xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c209920) : !flow.dispatch.tensor<readonly:tensor<9216x4096xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c667974912) : !flow.dispatch.tensor<readonly:tensor<4x4096xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c36864) : !flow.dispatch.tensor<writeonly:tensor<4x4096xf32>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 9216], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x9216xf32>> -> tensor<4x9216xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [9216, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<9216x4096xf32>> -> tensor<9216x4096xf32>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x4096xf32>> -> tensor<4x4096xf32>
        %8 = linalg.matmul ins(%4, %5 : tensor<4x9216xf32>, tensor<9216x4096xf32>) outs(%6 : tensor<4x4096xf32>) -> tensor<4x4096xf32>
        flow.dispatch.tensor.store %8, %3, offsets = [0, 0], sizes = [4, 4096], strides = [1, 1] : tensor<4x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x4096xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 128, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize, {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
//      CHECK: hal.executable.export public @matmul_4x4096x9216
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [32 : index, 4 : index, 1 : index]
//      CHECK: func.func @matmul_4x4096x9216()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Matvec does not go down matmul pipelines.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @matmul_1x4096x9216 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Shader, GroupNonUniform, GroupNonUniformShuffle], []>, NVIDIA:DiscreteGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 32>>
    }>) {
    hal.executable.export @matmul_1x4096x9216 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_1x4096x9216() {
        %c36864 = arith.constant 36864 : index
        %c667974912 = arith.constant 667974912 : index
        %c209920 = arith.constant 209920 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x9216xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c209920) : !flow.dispatch.tensor<readonly:tensor<9216x4096xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c667974912) : !flow.dispatch.tensor<readonly:tensor<1x4096xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c36864) : !flow.dispatch.tensor<writeonly:tensor<1x4096xf32>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 9216], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x9216xf32>> -> tensor<1x9216xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [9216, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<9216x4096xf32>> -> tensor<9216x4096xf32>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x4096xf32>> -> tensor<1x4096xf32>
        %8 = linalg.matmul ins(%4, %5 : tensor<1x9216xf32>, tensor<9216x4096xf32>) outs(%6 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
        flow.dispatch.tensor.store %8, %3, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : tensor<1x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x4096xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 2048], [1, 8], [0, 0, 8]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @matmul_1x4096x9216
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [256 : index, 1 : index, 1 : index]
//      CHECK: func.func @matmul_1x4096x9216()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Multi-reduction-dimension transposed-B matmul.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @multi_reduction_transposed_b_matmul {
  hal.executable.variant public @vulkan_spirv_fb target(#hal.executable.target<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Shader], []>, NVIDIA:DiscreteGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 32>>
    }>) {
    hal.executable.export public @multi_reduction_transposed_b_matmul layout(#pipeline_layout)
    builtin.module {
      func.func @multi_reduction_transposed_b_matmul() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x128xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x86x128xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4096x2048xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4096, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x128xf32>> -> tensor<4096x86x128xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2048, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x86x128xf32>> -> tensor<2048x86x128xf32>
          %5 = tensor.empty() : tensor<4096x2048xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4096x2048xf32>) -> tensor<4096x2048xf32>
          %7 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel", "reduction", "reduction"]
          } ins(%3, %4 : tensor<4096x86x128xf32>, tensor<2048x86x128xf32>) outs(%6 : tensor<4096x2048xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %8 = arith.mulf %in, %in_0 : f32
            %9 = arith.addf %out, %8 : f32
            linalg.yield %9 : f32
          } -> tensor<4096x2048xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [4096, 2048], strides = [1, 1] : tensor<4096x2048xf32> -> !flow.dispatch.tensor<writeonly:tensor<4096x2048xf32>>
          return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 128, 1, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize, {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
//      CHECK: hal.executable.export public @multi_reduction_transposed_b_matmul
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [32 : index, 8 : index, 1 : index]
//      CHECK: func.func @multi_reduction_transposed_b_matmul()
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]
