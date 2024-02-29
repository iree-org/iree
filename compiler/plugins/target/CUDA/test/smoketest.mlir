// RUN: iree-opt --split-input-file --iree-hal-transformation-pipeline %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-hal-transformation-pipeline --iree-hal-cuda-dump-ptx %s 2>&1 | FileCheck %s --check-prefix=PTX

#map = affine_map<(d0) -> (d0)>

module attributes {
  hal.device.targets = [
    #hal.device.target<"cuda", [
      #hal.executable.target<"cuda", "cuda-nvptx-fb">
    ]>
  ]
} {

stream.executable public @add_dispatch_0 {
  stream.executable.export @add_dispatch_0 workgroups(%arg0 : index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module  {
    func.func @add_dispatch_0(%arg0_binding: !stream.binding, %arg1_binding: !stream.binding, %arg2_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %arg0 = stream.binding.subspan %arg0_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<16xf32>>
      %arg1 = stream.binding.subspan %arg1_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<16xf32>>
      %arg2 = stream.binding.subspan %arg2_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<16xf32>>
      %0 = tensor.empty() : tensor<16xf32>
      %1 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
      %2 = flow.dispatch.tensor.load %arg1, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
      %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %2 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> tensor<16xf32>
      flow.dispatch.tensor.store %3, %arg2, offsets=[0], sizes=[16], strides=[1] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:tensor<16xf32>>
      return
    }
  }
}

}

// PTX: .entry add_dispatch_0
// PTX: .maxntid 64, 1, 1
// PTX:   add.rn.f32

//      CHECK:   hal.executable.binary public @cuda_nvptx_fb attributes {
// CHECK-SAME:     data = dense
// CHECK-SAME:     format = "cuda-nvptx-fb"
