// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true})))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @matmul_1x4096x9216 {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Shader], []>, NVIDIA:DiscreteGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 32>>
    }> {
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

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 256], [1, 4], [0, 0, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize pipeline_depth = 1>
//      CHECK: hal.executable.export public @matmul_1x4096x9216
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [64 : index, 1 : index, 1 : index]
//      CHECK: func.func @matmul_1x4096x9216()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]
