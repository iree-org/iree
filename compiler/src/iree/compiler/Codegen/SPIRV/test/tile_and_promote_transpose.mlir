// RUN: tools/iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-alloc)))))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32,32]]>

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
hal.executable @shared_mem_transpose {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader], []>, AMD:DiscreteGPU, #spirv.resource_limits<
      max_compute_shared_memory_size = 65536,
      max_compute_workgroup_invocations = 1024,
      max_compute_workgroup_size = [1024, 1024, 1024],
      subgroup_size = 64>>}> {
    hal.executable.export public @shared_mem_transpose ordinal(0) layout(#pipeline_layout) attributes {
      workgroup_size = [2 : index, 64 : index, 1 : index]
    }
    builtin.module {
      func.func @shared_mem_transpose() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<2048x768xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<768x2048xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x768xf32>> -> tensor<2048x768xf32>
        %3 = tensor.empty() : tensor<768x2048xf32>
        %4 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<2048x768xf32>) outs(%3 : tensor<768x2048xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          linalg.yield %arg0 : f32
        } -> tensor<768x2048xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [768, 2048], strides = [1, 1] : tensor<768x2048xf32> -> !flow.dispatch.tensor<writeonly:tensor<768x2048xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @shared_mem_transpose()

