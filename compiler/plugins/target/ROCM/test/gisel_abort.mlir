// RUN: not not iree-opt --iree-hip-llvm-global-isel --iree-hal-transformation-pipeline --iree-gpu-test-target=gfx90a %s 2>&1  | FileCheck %s
// For some reason, 2 `not`s are required to convert the abort error code from 134 to 0

module attributes {
  hal.device.targets = [
    #hal.device.target<"amdgpu", [
      #hal.executable.target<"rocm", "amdgcn-amd-amdhsa">
    ]> : !hal.device
  ]
} {

stream.executable public @add_dispatch_executable {
  stream.executable.export @add_dispatch workgroups(%arg0 : index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module  {
    func.func @add_dispatch(%arg0_binding: !stream.binding, %arg1_binding: !stream.binding, %arg2_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %arg0 = stream.binding.subspan %arg0_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<16xi32>>
      %arg1 = stream.binding.subspan %arg1_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<16xi32>>
      %arg2 = stream.binding.subspan %arg2_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<16xi32>>
      %0 = tensor.empty() : tensor<16xi32>
      %1 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xi32>> -> tensor<16xi32>
      %2 = flow.dispatch.tensor.load %arg1, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xi32>> -> tensor<16xi32>
      %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %2 : tensor<16xi32>, tensor<16xi32>) outs(%0 : tensor<16xi32>) {
      ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
        // CHECK: LLVM ERROR: unable to map instruction: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.x86.wrpkru)
        // CHECK: llvm::reportGISelFailure
        llvm.call_intrinsic "llvm.x86.wrpkru"(%arg4) : (i32) -> ()
        %4 = arith.addi %arg3, %arg4 : i32
        linalg.yield %4 : i32
      } -> tensor<16xi32>
      flow.dispatch.tensor.store %3, %arg2, offsets=[0], sizes=[16], strides=[1] : tensor<16xi32> -> !flow.dispatch.tensor<writeonly:tensor<16xi32>>
      return
    }
  }
}
}
