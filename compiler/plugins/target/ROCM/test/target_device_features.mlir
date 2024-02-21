// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targets=rocm},iree-hal-transformation-pipeline{serialize-executables=false})' --iree-rocm-target-chip=gfx940 %s | FileCheck %s --check-prefix=MI300
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targets=rocm},iree-hal-transformation-pipeline{serialize-executables=false})' --iree-rocm-target-chip=gfx942 %s | FileCheck %s --check-prefix=MI300

// MI300: mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>]

stream.executable public @reduce_dispatch {
  stream.executable.export @reduce_dispatch workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @reduce_dispatch(%arg0_binding: !stream.binding, %arg1_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %arg0 = stream.binding.subspan %arg0_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<16xf32>>
      %arg1 = stream.binding.subspan %arg1_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<f32>>
      %0 = tensor.empty() : tensor<f32>
      %1 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
      %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%1 : tensor<16xf32>) outs(%0 : tensor<f32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %4 = arith.addf %arg2, %arg3 : f32
        linalg.yield %4 : f32
      } -> tensor<f32>
      flow.dispatch.tensor.store %3, %arg1, offsets=[], sizes=[], strides=[] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
      return
    }
  }
}
