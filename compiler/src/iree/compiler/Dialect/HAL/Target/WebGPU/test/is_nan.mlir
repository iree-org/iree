// RUN: iree-opt --split-input-file --iree-hal-transformation-pipeline \
// RUN:   --verify-diagnostics %s

// RUN: iree-opt --split-input-file --iree-hal-transformation-pipeline \
// RUN:   --verify-diagnostics %s 2>&1 | FileCheck %s

// This test generates the 'isNan' function, which fails to compile.
// TODO(#10142): Fix compilation and test success (fold into smoketest.mlir?)
//     (or delete - operation coverage should probably use tests/e2e/ instead)

#map0 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {
  hal.device.targets = [
    #hal.device.target<"webgpu", {
      executable_targets = [
        #hal.executable.target<"webgpu-wgsl", "webgpu-wgsl-fb", {
          spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spv.resource_limits<>>
        }>
      ]
    }>
  ]
} {

// CHECK:      Tint reported 2 error(s) for a SPIR-V program, see diagnostics:
// CHECK-NEXT: error: unknown function: 'isNan'
// CHECK-NEXT: error: unknown function: 'isNan'
//
// expected-error @+3 {{failed to compile SPIR-V to WGSL}}
// expected-error @+2 {{failed to serialize executable for target backend webgpu-wgsl}}
// expected-error @+1 {{failed to serialize executables}}
stream.executable public @min_dispatch {
  stream.executable.export public @min_dispatch workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.default_workgroup_count %arg0, %arg1
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @min_dispatch(%arg0: !stream.binding, %arg1: !stream.binding) {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0x7FC00000 : f32
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:1x5xf32>
      %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readwrite:1x5xf32>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 5], strides = [1, 1] : !flow.dispatch.tensor<readonly:1x5xf32> -> tensor<1x5xf32>
      %3 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, 5], strides = [1, 1] : !flow.dispatch.tensor<readwrite:1x5xf32> -> tensor<1x5xf32>
      %4 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<1x5xf32>) outs(%3 : tensor<1x5xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %5 = arith.cmpf ogt, %arg2, %arg3 : f32
        %6 = arith.select %5, %arg2, %arg3 : f32
        %7 = arith.cmpf uno, %arg2, %arg3 : f32
        %8 = arith.select %7, %cst, %6 : f32
        linalg.yield %8 : f32
      } -> tensor<1x5xf32>
      flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [1, 5], strides = [1, 1] : tensor<1x5xf32> -> !flow.dispatch.tensor<readwrite:1x5xf32>
      return
    }
  }
}

}
