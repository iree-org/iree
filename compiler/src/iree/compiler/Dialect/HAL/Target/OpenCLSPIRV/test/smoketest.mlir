// RUN: iree-opt --split-input-file --iree-hal-transformation-pipeline %s | FileCheck %s

module attributes {
  hal.device.targets = [
    #hal.device.target<"vulkan", {
      executable_targets = [
        #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
          spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spv.resource_limits<>>
        }>
      ]
    }>
  ]
} {

stream.executable public @reduce_dispatch {
  stream.executable.export @reduce_dispatch
  builtin.module {
    func.func @reduce_dispatch(%arg0_binding: !stream.binding, %arg1_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %arg0 = stream.binding.subspan %arg0_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:16xf32>
      %arg1 = stream.binding.subspan %arg1_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:f32>
      %0 = linalg.init_tensor [] : tensor<f32>
      %1 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%1 : tensor<16xf32>) outs(%0 : tensor<f32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %4 = arith.addf %arg2, %arg3 : f32
        linalg.yield %4 : f32
      } -> tensor<f32>
      flow.dispatch.tensor.store %3, %arg1, offsets=[], sizes=[], strides=[] : tensor<f32> -> !flow.dispatch.tensor<writeonly:f32>
      return
    }
  }
}

}

//      CHECK:   hal.executable.binary public @vulkan_spirv_fb attributes
// CHECK-SAME:     data = dense
// CHECK-SAME:     format = "vulkan-spirv-fb"
