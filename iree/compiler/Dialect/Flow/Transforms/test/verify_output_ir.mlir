// RUN: iree-opt -iree-verify-output-legality -verify-diagnostics %s -split-input-file

func @check_only_flow_uses_tensor_types(%arg0: tensor<i1>) -> tensor<i8> {
  // expected-error @+1 {{illegal operation returning 'tensor<i8>' in output from flow transformation, flow dialect ops should be used for acting on tensors at this point}}
  %0 = zexti %arg0 : tensor<i1> to tensor<i8>
  return %0 : tensor<i8>
}

// -----

func @flow_can_use_tensor_types(%arg0: !hal.buffer_view) -> !hal.buffer_view {
  %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<i32>
  %1 = flow.tensor.load %0 : tensor<i32>
  %2 = flow.ex.stream.fragment(%1) : (i32) -> tensor<i32> =
      (%arg1: i32) -> tensor<i32> {
    %3 = flow.tensor.splat %arg1 : tensor<i32>
    flow.return %3 : tensor<i32>
  }
  %3 = hal.tensor.cast %2 : tensor<i32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}
