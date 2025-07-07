// RUN: iree-opt --pass-pipeline="builtin.module(torch-iree-func-conversion)" --allow-unregistered-dialect --split-input-file %s | FileCheck %s --check-prefix=CHECK-ASYNC

// -----
// Tests the immutable + mutable argument case where the mutable argument is
// overwritten as part of the function cleanup and the argument is not returned.
// This exhaustively verifies the async function.
// Note that the order of the barrier operands and successors is implementation
// dependent, and the current implementation processes mutable before
// immutable.
// CHECK-ASYNC-LABEL: @mutable_input_overwrite_no_return
//       CHECK-ASYNC: util.func public @main$async(
//  CHECK-ASYNC-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view,
//  CHECK-ASYNC-SAME:     %arg2: !hal.fence, %arg3: !hal.fence) -> !hal.buffer_view
//   CHECK-ASYNC-DAG: %[[WAIT_ARG0:.+]] = hal.tensor.import wait(%arg2) => %arg0
//   CHECK-ASYNC-DAG: %[[TORCH_ARG0:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG0]]
//   CHECK-ASYNC-DAG: %[[WAIT_ARG1:.+]] = hal.tensor.import wait(%arg2) => %arg1
//   CHECK-ASYNC-DAG: %[[TORCH_ARG1:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG1]]
//   CHECK-ASYNC-DAG: %[[TORCH_RESULT0:.+]] = torch.operator "other_calc"(%[[TORCH_ARG0]])
//   CHECK-ASYNC-DAG: %[[TORCH_RESULT1:.+]] = torch.operator "mutate_inplace"(%[[TORCH_ARG1]])
//   CHECK-ASYNC-DAG: %[[TENSOR_ARG0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-ASYNC-DAG: %[[TENSOR_ARG1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//       CHECK-ASYNC: %[[EXPORT_ALIAS1:.+]] = hal.tensor.alias wait(%arg2) => %[[TENSOR_ARG1]] : tensor<5x4xf32> to %arg1 : !hal.buffer_view
//       CHECK-ASYNC: %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[EXPORT_ALIAS1]], %[[TENSOR_ARG0]] : tensor<5x4xf32>, tensor<4x5xi32>) => %arg3 : !hal.fence
//   CHECK-ASYNC-DAG: %[[EXPORT_RESULT0:.+]] = hal.tensor.export %[[BARRIER_RESULTS]]#0
//   CHECK-ASYNC-DAG: %[[EXPORT_RESULT1:.+]] = hal.tensor.export %[[BARRIER_RESULTS]]#1
//       CHECK-ASYNC: util.return %[[EXPORT_RESULT1]]
builtin.module @mutable_input_overwrite_no_return {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>, %arg1: !torch.tensor<[5,4],f32>)
    -> (!torch.vtensor<[4,5],si32>) {
  %0 = torch.copy.to_vtensor %arg1 : !torch.vtensor<[5,4],f32>
  %1 = torch.operator "mutate_inplace"(%0) : (!torch.vtensor<[5,4],f32>) -> !torch.vtensor<[5,4],f32>
  %2 = torch.operator "other_calc"(%arg0) : (!torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>
  torch.overwrite.tensor.contents %1 overwrites %arg1 : !torch.vtensor<[5,4],f32>, !torch.tensor<[5,4],f32>
  return %2 : !torch.vtensor<[4,5],si32>
}
}

// -----
// This isn't a great program to write but is legal. It verifies that if the
// function returns an intermediate vtensor just before it was noted as mutated
// that we export it properly. This would be a hard program to write in PyTorch
// but possible to end up this way so testing the corner.
// Not a good idea to do but legal. This verifies that if returning a mutated
// tensor's intermediate value, you will get two exports, indicating a copy.
// CHECK-ASYNC-LABEL: @mutable_input_overwrite_return_alias_copies
//       CHECK-ASYNC: %[[ALIASED:.+]] = hal.tensor.alias wait({{.+}}) => %{{.+}} : tensor<5x4xf32> to %arg0 : !hal.buffer_view
//       CHECK-ASYNC: %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[ALIASED]], %{{.*}} : tensor<5x4xf32>, tensor<5x4xf32>)
//   CHECK-ASYNC-DAG: = hal.tensor.export %[[BARRIER_RESULTS]]#0
//   CHECK-ASYNC-DAG: = hal.tensor.export %[[BARRIER_RESULTS]]#1
builtin.module @mutable_input_overwrite_return_alias_copies {
func.func @main(%arg0: !torch.tensor<[5,4],f32>) -> (!torch.vtensor<[5,4],f32>) {
  %0 = torch.copy.to_vtensor %arg0 : !torch.vtensor<[5,4],f32>
  %1 = torch.operator "mutate_inplace"(%0) : (!torch.vtensor<[5,4],f32>) -> !torch.vtensor<[5,4],f32>
  torch.overwrite.tensor.contents %1 overwrites %arg0 : !torch.vtensor<[5,4],f32>, !torch.tensor<[5,4],f32>
  return %1 : !torch.vtensor<[5,4],f32>
}
}

// -----
// Tests the immutable + mutable argument case with explicit affinities.
// CHECK-ASYNC-LABEL: @mutable_input_overwrite_no_return
//       CHECK-ASYNC: util.func public @main$async(
//  CHECK-ASYNC-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view,
//  CHECK-ASYNC-SAME:     %arg2: !hal.fence, %arg3: !hal.fence) -> !hal.buffer_view
//   CHECK-ASYNC-DAG: %[[WAIT_ARG0:.+]] = hal.tensor.import on(#hal.device.promise<@dev_a>) wait(%arg2) => %arg0
//   CHECK-ASYNC-DAG: %[[TORCH_ARG0:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG0]]
//   CHECK-ASYNC-DAG: %[[WAIT_ARG1:.+]] = hal.tensor.import on(#hal.device.promise<@dev_b>) wait(%arg2) => %arg1
//   CHECK-ASYNC-DAG: %[[TORCH_ARG1:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG1]]
//   CHECK-ASYNC-DAG: %[[TORCH_RESULT0:.+]] = torch.operator "other_calc"(%[[TORCH_ARG0]])
//   CHECK-ASYNC-DAG: %[[TORCH_RESULT1:.+]] = torch.operator "mutate_inplace"(%[[TORCH_ARG1]])
//   CHECK-ASYNC-DAG: %[[TENSOR_ARG0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-ASYNC-DAG: %[[TENSOR_ARG1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//       CHECK-ASYNC: %[[EXPORT_ALIAS1:.+]] = hal.tensor.alias on(#hal.device.promise<@dev_b>) wait(%arg2) => %[[TENSOR_ARG1]] : tensor<5x4xf32> to %arg1 : !hal.buffer_view
//       CHECK-ASYNC: %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[EXPORT_ALIAS1]], %[[TENSOR_ARG0]] : tensor<5x4xf32>, tensor<4x5xi32>) => %arg3 : !hal.fence
//   CHECK-ASYNC-DAG: %[[EXPORT_RESULT0:.+]] = hal.tensor.export on(#hal.device.promise<@dev_b>) %[[BARRIER_RESULTS]]#0
//   CHECK-ASYNC-DAG: %[[EXPORT_RESULT1:.+]] = hal.tensor.export on(#hal.device.promise<@dev_a>) %[[BARRIER_RESULTS]]#1
//       CHECK-ASYNC: util.return %[[EXPORT_RESULT1]]
builtin.module @mutable_input_overwrite_no_return_affinities {
func.func @main(%arg0: !torch.vtensor<[4,5],si32> {iree.abi.affinity = #hal.device.promise<@dev_a>},
                %arg1: !torch.tensor<[5,4],f32> {iree.abi.affinity = #hal.device.promise<@dev_b>})
    -> (!torch.vtensor<[4,5],si32> {iree.abi.affinity = #hal.device.promise<@dev_a>}) {
  %0 = torch.copy.to_vtensor %arg1 : !torch.vtensor<[5,4],f32>
  %1 = torch.operator "mutate_inplace"(%0) : (!torch.vtensor<[5,4],f32>) -> !torch.vtensor<[5,4],f32>
  %2 = torch.operator "other_calc"(%arg0) : (!torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>
  torch.overwrite.tensor.contents %1 overwrites %arg1 : !torch.vtensor<[5,4],f32>, !torch.tensor<[5,4],f32>
  return %2 : !torch.vtensor<[4,5],si32>
}
}
