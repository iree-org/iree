// RUN: iree-opt --split-input-file --iree-hal-transformation-pipeline \
// RUN:   --verify-diagnostics %s

// RUN: iree-opt --split-input-file --iree-hal-transformation-pipeline \
// RUN:   --verify-diagnostics %s 2>&1 | FileCheck %s

// This test uses 'readwrite' memory, which fails to compile.
// TODO(#10145): Fix compilation and test success (fold into smoketest.mlir?)

module attributes {
  hal.device.targets = [
    #hal.device.target<"webgpu", {
      executable_targets = [
        #hal.executable.target<"webgpu-wgsl", "webgpu-wgsl-fb", {
          spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
        }>
      ]
    }>
  ]
} {

// CHECK:      Tint reported 1 error(s) for a SPIR-V program, see diagnostics:
// CHECK-NEXT: error: cannot initialize let of type 'ptr<storage, f32, read>' with value of type 'ptr<storage, f32, read_write>'
//
// expected-error @+3 {{failed to compile SPIR-V to WGSL}}
// expected-error @+2 {{failed to serialize executable for target backend webgpu-wgsl}}
// expected-error @+1 {{failed to serialize executables}}
stream.executable public @reduce_dispatch {
  stream.executable.export @reduce_dispatch workgroups(%arg0 : index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @reduce_dispatch(%arg0_binding: !stream.binding, %arg1_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %arg0 = stream.binding.subspan %arg0_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:16xf32>
      %arg1 = stream.binding.subspan %arg1_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:f32>
      %0 = linalg.init_tensor [] : tensor<f32>
      %1 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%1 : tensor<16xf32>) outs(%0 : tensor<f32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %4 = arith.addf %arg2, %arg3 : f32
        linalg.yield %4 : f32
      } -> tensor<f32>
      flow.dispatch.tensor.store %2, %arg1, offsets=[], sizes=[], strides=[] : tensor<f32> -> !flow.dispatch.tensor<writeonly:f32>
      return
    }
  }
}

}
