// RUN: iree-opt %s --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(transform-preload-library{transform-library-paths=%p/transform_dialect_dummy_spec.mlir},iree-spirv-lower-executable-target-pass)))" | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @copy_f32 {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformShuffle], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
       subgroup_size = 16>>
    }> {
    hal.executable.export public @copy_f32 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // CHECK: IR printer:
      func.func @copy_f32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x2xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x2xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %3 = tensor.empty() : tensor<2x2xf32>
        %4 = linalg.generic {
            indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
            ins(%2 : tensor<2x2xf32>) outs(%3 : tensor<2x2xf32>) {
          ^bb0(%arg0: f32, %arg1: f32):
            %5 = math.sqrt %arg0 : f32
            linalg.yield %5 : f32
          } -> tensor<2x2xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : tensor<2x2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x2xf32>>
        return
      }
    }
    // CHECK-COUNT-2: vector.transfer_read
    // CHECK-COUNT-2: math.sqrt
    // CHECK-COUNT-2: vector.transfer_write
  }
}
