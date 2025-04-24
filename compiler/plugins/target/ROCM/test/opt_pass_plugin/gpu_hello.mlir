// RUN: iree-opt --split-input-file --iree-hal-transformation-pipeline --iree-gpu-test-target=gfx90a \
// RUN:   --iree-hip-pass-plugin-path=$IREE_BINARY_DIR/lib/libGPUHello$IREE_DYLIB_EXT %s 2>&1 | FileCheck %s

module attributes {
  hal.device.targets = [
    #hal.device.target<"hip", [
      #hal.executable.target<"rocm", "rocm-hsaco-fb">
    ]> : !hal.device
  ]
} {

stream.executable public @add_dispatch_0 {
  stream.executable.export @add_dispatch_0 workgroups(%arg0 : index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module  {
    func.func @add_dispatch_0(%arg0_binding: !stream.binding, %arg1_binding: !stream.binding, %arg2_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %arg0 = stream.binding.subspan %arg0_binding[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>>
      %arg1 = stream.binding.subspan %arg1_binding[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>>
      %arg2 = stream.binding.subspan %arg2_binding[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
      %0 = tensor.empty() : tensor<16xf32>
      %1 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets=[0], sizes=[16], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
      %2 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets=[0], sizes=[16], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
      %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %2 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> tensor<16xf32>
      iree_tensor_ext.dispatch.tensor.store %3, %arg2, offsets=[0], sizes=[16], strides=[1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
      return
    }
  }
}

}

// CHECK: Hello From First Instruction of GPU Kernel: add_dispatch_0
