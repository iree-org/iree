// RUN: iree-opt -split-input-file -iree-hal-transformation-pipeline %s | IreeFileCheck %s

module attributes {
  hal.device.targets = [
    #hal.device.target<"rocm", {
      executable_targets = [
        #hal.executable.target<"rocm", "rocm-hsaco-fb">
      ]
    }>
  ]
} {

stream.executable public @add_dispatch_0 {
  stream.executable.export @add_dispatch_0
  builtin.module  {
    func @add_dispatch_0(%arg0_binding: !stream.binding, %arg1_binding: !stream.binding, %arg2_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %arg0 = stream.binding.subspan %arg0_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:16xf32>
      %arg1 = stream.binding.subspan %arg1_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:16xf32>
      %arg2 = stream.binding.subspan %arg2_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:16xf32>
      %0 = linalg.init_tensor [16] : tensor<16xf32>
      %1 = flow.dispatch.tensor.load %arg0, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %2 = flow.dispatch.tensor.load %arg1, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %2 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> tensor<16xf32>
      flow.dispatch.tensor.store %3, %arg2, offsets=[], sizes=[], strides=[] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:16xf32>
      return
    }
  }
}

}

//      CHECK:   hal.executable.binary public @rocm_hsaco_fb attributes {
// CHECK-SAME:     data = dense
// CHECK-SAME:     format = "rocm-hsaco-fb"
