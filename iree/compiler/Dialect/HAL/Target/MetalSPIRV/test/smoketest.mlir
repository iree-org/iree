// RUN: iree-opt -split-input-file -iree-hal-transformation-pipeline %s | IreeFileCheck %s

module attributes {
  hal.device.targets = [
    #hal.device.target<"metal", {
      executable_targets = [
        #hal.executable.target<"metal-spirv", "metal-msl-fb", {
          spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
        }>
      ]
    }>
  ]
} {

flow.executable @reduce_dispatch {
  flow.dispatch.entry @reduce_dispatch attributes {workgroup_rank = 3 : index}
  builtin.module {
    func @reduce_dispatch(%arg0: !flow.dispatch.tensor<readonly:16xf32>, %arg1: !flow.dispatch.tensor<writeonly:f32>) {
      %0 = linalg.init_tensor [] : tensor<f32>
      %1 = flow.dispatch.tensor.load %arg0, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
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

// CHECK:        hal.executable.binary public @metal_msl_fb attributes {
// CHECK-SAME:     data = dense
// CHECK-SAME:     format = "metal-msl-fb"
