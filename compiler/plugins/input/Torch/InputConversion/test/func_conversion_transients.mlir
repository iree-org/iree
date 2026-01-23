// RUN: iree-opt --pass-pipeline="builtin.module(torch-iree-func-conversion{externalize-transients=true})" --allow-unregistered-dialect --split-input-file %s | FileCheck %s

// This file contains some tests from the parallel file `func_conversion.mlir`, but is run with `externalize-transients=True`.

// CHECK-LABEL: @immutable_import_export
//       CHECK: util.func public @main$async(
//  CHECK-SAME:     %[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view,
//  CHECK-SAME:     %[[TRANSIENT_STORAGE:.+]]: !hal.buffer,
//  CHECK-SAME:     %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence) ->
//  CHECK-SAME:     (!hal.buffer_view, !hal.buffer_view)
//  CHECK-SAME:     iree.abi.model = "coarse-fences"
//  CHECK-SAME:     iree.abi.stub
//   CHECK-DAG:   %[[WAIT_ARG0:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[ARG0]] : !hal.buffer_view -> tensor<4x5xi32>
//   CHECK-DAG:   %[[WAIT_ARG1:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[ARG1]] : !hal.buffer_view -> tensor<5x4xf32>
//   CHECK-DAG:   %[[TORCH_ARG0:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG0]] : tensor<4x5xi32> -> !torch.vtensor<[4,5],si32>
//   CHECK-DAG:   %[[TORCH_ARG1:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG1]] : tensor<5x4xf32> -> !torch.vtensor<[5,4],f32>
//   CHECK-DAG:   %[[TORCH_RESULT0:.+]] = torch.operator "foobar0"(%[[TORCH_ARG0]])
//   CHECK-DAG:   %[[TORCH_RESULT1:.+]] = torch.operator "foobar1"(%[[TORCH_ARG1]])
//   CHECK-DAG:   %[[TENSOR_RESULT0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-DAG:   %[[TENSOR_RESULT1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//   CHECK-DAG:   %[[TRANSIENT_CALL0:.+]] = hal.tensor.transients %[[TENSOR_RESULT0]] : tensor<4x5xi32> from %[[TRANSIENT_STORAGE]] : !hal.buffer
//   CHECK-DAG:   %[[TRANSIENT_CALL1:.+]] = hal.tensor.transients %[[TENSOR_RESULT1]] : tensor<5x4xf32> from %[[TRANSIENT_STORAGE]] : !hal.buffer
//       CHECK:   %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[TRANSIENT_CALL0]], %[[TRANSIENT_CALL1]] : tensor<4x5xi32>, tensor<5x4xf32>) => %[[SIGNAL]]
//   CHECK-DAG:   %[[FUNC_RESULT0:.+]] = hal.tensor.export %[[BARRIER_RESULTS]]#0
//   CHECK-DAG:   %[[FUNC_RESULT1:.+]] = hal.tensor.export %[[BARRIER_RESULTS]]#1
//       CHECK:   util.return %[[FUNC_RESULT0]], %[[FUNC_RESULT1]]
//
//       CHECK: util.func public @main(%[[SYNC_ARG0:.+]]: !hal.buffer_view, %[[SYNC_ARG1:.+]]: !hal.buffer_view, %[[SYNC_STORAGE:.+]]: !hal.buffer)
//  CHECK-SAME:     -> (!hal.buffer_view, !hal.buffer_view)
//  CHECK-SAME:     iree.abi.stub
//   CHECK-DAG:   %[[CONSTANT0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[CONSTANT1:.+]] = arith.constant -1 : i32
//   CHECK-DAG:   %[[DEVICE0:.+]] = hal.devices.get %[[CONSTANT0]] : !hal.device
//   CHECK-DAG:   %[[NULL_FENCE:.+]] = util.null : !hal.fence
//       CHECK:   %[[NEW_FENCE:.+]] = hal.fence.create device(%[[DEVICE0]] : !hal.device) flags("None")
//       CHECK:   %[[CALL_RESULTS:.+]]:2 = util.call @main$async(%[[SYNC_ARG0]], %[[SYNC_ARG1]], %[[SYNC_STORAGE]], %[[NULL_FENCE]], %[[NEW_FENCE]])
//       CHECK:   %[[AWAIT_STATUS:.+]] = hal.fence.await until([%[[NEW_FENCE]]]) timeout_millis(%[[CONSTANT1]])
//       CHECK:   util.return %[[CALL_RESULTS]]#0, %[[CALL_RESULTS]]#1 : !hal.buffer_view, !hal.buffer_view
builtin.module @immutable_import_export {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>, %arg1: !torch.vtensor<[5,4],f32>)
    -> (!torch.vtensor<[4,5],si32>, !torch.vtensor<[5,4],f32>) {
  %0 = torch.operator "foobar0"(%arg0) : (!torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>
  %1 = torch.operator "foobar1"(%arg1) : (!torch.vtensor<[5,4],f32>) -> !torch.vtensor<[5,4],f32>
  return %0, %1 : !torch.vtensor<[4,5],si32>, !torch.vtensor<[5,4],f32>
}
}

// -----
// Converting this trivial return func should still generate the external buffer arg.
// However, there should not be any `!hal.tensor.transients` op created.
// CHECK-LABEL:   @return_immutable_arg
//       CHECK:   util.func public @main$async(%[[ARG0:.+]]: !hal.buffer_view,
//  CHECK-SAME:       %[[EXTERNAL_BUFFER:.+]]: !hal.buffer,
//  CHECK-SAME:       %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence) ->
//       CHECK:   hal.fence.signal<%[[SIGNAL]] : !hal.fence>
//       CHECK:   util.return %[[ARG0]]
//       CHECK:   util.func public @main(%[[SYNC_ARG0:.+]]: !hal.buffer_view, %[[SYNC_EXTERNAL_BUFFER:.+]]: !hal.buffer)
//   CHECK-NOT:   hal.tensor.transients
builtin.module @return_immutable_arg {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>  {
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-LABEL: @mutable_input_overwrite_no_return
//       CHECK: util.func public @main$async(
//  CHECK-SAME:     %[[INPUT:.+]]: !hal.buffer_view, %[[MUTABLE_INPUT:.+]]: !hal.buffer_view,
//  CHECK-SAME:     %[[EXTERNAL_BUFFER:.+]]: !hal.buffer,
//  CHECK-SAME:     %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence) -> !hal.buffer_view
//   CHECK-DAG: %[[WAIT_INPUT:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[INPUT]]
//   CHECK-DAG: %[[TORCH_INPUT:.+]] = torch_c.from_builtin_tensor %[[WAIT_INPUT]]
//   CHECK-DAG: %[[WAIT_MUTABLE_INPUT:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[MUTABLE_INPUT]]
//   CHECK-DAG: %[[TORCH_MUTABLE_INPUT:.+]] = torch_c.from_builtin_tensor %[[WAIT_MUTABLE_INPUT]]
//   CHECK-DAG: %[[TORCH_RESULT0:.+]] = torch.operator "other_calc"(%[[TORCH_INPUT]])
//   CHECK-DAG: %[[TORCH_RESULT1:.+]] = torch.operator "mutate_inplace"(%[[TORCH_MUTABLE_INPUT]])
//   CHECK-DAG: %[[TENSOR_RESULT0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-DAG: %[[TENSOR_RESULT1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//       CHECK: %[[ALIAS:.+]] = hal.tensor.alias wait(%[[WAIT]]) => %[[TENSOR_RESULT1]] : tensor<5x4xf32> to %[[MUTABLE_INPUT]] : !hal.buffer_view
//   CHECK-DAG: %[[TRANSIENTS1:.+]] = hal.tensor.transients %[[ALIAS]] : tensor<5x4xf32> from %[[EXTERNAL_BUFFER]] : !hal.buffer
//   CHECK-DAG: %[[TRANSIENTS0:.+]] = hal.tensor.transients %[[TENSOR_RESULT0]] : tensor<4x5xi32> from %[[EXTERNAL_BUFFER]] : !hal.buffer
//       CHECK: %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[TRANSIENTS1]], %[[TRANSIENTS0]] : tensor<5x4xf32>, tensor<4x5xi32>) => %[[SIGNAL]] : !hal.fence
//   CHECK-DAG: %[[EXPORT_RESULT0:.+]] = hal.tensor.export %[[BARRIER_RESULTS]]#0
//   CHECK-DAG: %[[EXPORT_RESULT1:.+]] = hal.tensor.export %[[BARRIER_RESULTS]]#1
//       CHECK: util.return %[[EXPORT_RESULT1]]
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
// This should be generating `hal.tensor.transients` with dynamic dims.
// CHECK-LABEL:  @immutable_import_export {
//       CHECK:  util.func public @main$async(
//  CHECK-SAME:     %[[INPUT0:.+]]: !hal.buffer_view, %[[INPUT1:.+]]: !hal.buffer_view,
//  CHECK-SAME:     %[[EXTERNAL_BUFFER:.+]]: !hal.buffer, %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence)
//  CHECK-SAME:     -> (!hal.buffer_view, !hal.buffer_view)
//       CHECK:  %[[INPUT0_DIM1:.+]] = hal.buffer_view.dim<%[[INPUT0]] : !hal.buffer_view>[1] : index
//       CHECK:  %[[WAIT_INPUT0:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[INPUT0]] : !hal.buffer_view -> tensor<4x?xi32>{%[[INPUT0_DIM1]]}
//       CHECK:  %[[TORCH_INPUT0:.+]] = torch_c.from_builtin_tensor %[[WAIT_INPUT0]] : tensor<4x?xi32> -> !torch.vtensor<[4,?],si32>
//       CHECK:  %[[INPUT1_DIM0:.+]] = hal.buffer_view.dim<%[[INPUT1]] : !hal.buffer_view>[0] : index
//       CHECK:  %[[WAIT_INPUT1:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[INPUT1]] : !hal.buffer_view -> tensor<?x4xf32>{%[[INPUT1_DIM0]]}
//       CHECK:  %[[TORCH_INPUT1:.+]] = torch_c.from_builtin_tensor %[[WAIT_INPUT1]] : tensor<?x4xf32> -> !torch.vtensor<[?,4],f32>
//       CHECK:  %[[TORCH_RESULT0:.+]] = torch.operator "foobar0"(%[[TORCH_INPUT0]]) : (!torch.vtensor<[4,?],si32>) -> !torch.vtensor<[4,?],si32>
//       CHECK:  %[[TORCH_RESULT1:.+]] = torch.operator "foobar1"(%[[TORCH_INPUT1]]) : (!torch.vtensor<[?,4],f32>) -> !torch.vtensor<[?,4],f32>
//       CHECK:  %[[TENSOR_RESULT0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]] : !torch.vtensor<[4,?],si32> -> tensor<4x?xi32>
//       CHECK:  %[[TENSOR_RESULT1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]] : !torch.vtensor<[?,4],f32> -> tensor<?x4xf32>
//       CHECK:  %[[CST1:.+]] = arith.constant 1 : index
//       CHECK:  %[[RESULT0_DIM1:.+]] = tensor.dim %[[TENSOR_RESULT0]], %[[CST1]] : tensor<4x?xi32>
//       CHECK:  %[[TRANSIENTS0:.+]] = hal.tensor.transients %[[TENSOR_RESULT0]] : tensor<4x?xi32>{%[[RESULT0_DIM1]]} from %[[EXTERNAL_BUFFER]] : !hal.buffer
//       CHECK:  %[[CST0:.+]] = arith.constant 0 : index
//       CHECK:  %[[RESULT1_DIM0:.+]] = tensor.dim %[[TENSOR_RESULT1]], %[[CST0]] : tensor<?x4xf32>
//       CHECK:  %[[TRANSIENTS1:.+]] = hal.tensor.transients %[[TENSOR_RESULT1]] : tensor<?x4xf32>{%[[RESULT1_DIM0]]} from %[[EXTERNAL_BUFFER]] : !hal.buffer
//       CHECK:  %[[BARRIER:.+]]:2 = hal.tensor.barrier join(%[[TRANSIENTS0]], %[[TRANSIENTS1]] : tensor<4x?xi32>, tensor<?x4xf32>) => %[[SIGNAL]] : !hal.fence
//       CHECK:  %[[EXPORT0:.+]] = hal.tensor.export %[[BARRIER]]#0 : tensor<4x?xi32>{%[[RESULT0_DIM1]]} -> !hal.buffer_view
//       CHECK:  %[[EXPORT1:.+]] = hal.tensor.export %[[BARRIER]]#1 : tensor<?x4xf32>{%[[RESULT1_DIM0]]} -> !hal.buffer_view
//       CHECK:  util.return %[[EXPORT0]], %[[EXPORT1]] : !hal.buffer_view, !hal.buffer_view
builtin.module @immutable_import_export {
func.func @main(%arg0: !torch.vtensor<[4,?],si32>, %arg1: !torch.vtensor<[?,4],f32>)
    -> (!torch.vtensor<[4,?],si32>, !torch.vtensor<[?,4],f32>) {
  %0 = torch.operator "foobar0"(%arg0) : (!torch.vtensor<[4,?],si32>) -> !torch.vtensor<[4,?],si32>
  %1 = torch.operator "foobar1"(%arg1) : (!torch.vtensor<[?,4],f32>) -> !torch.vtensor<[?,4],f32>
  return %0, %1 : !torch.vtensor<[4,?],si32>, !torch.vtensor<[?,4],f32>
}
}

// -----
// CHECK-LABEL: @mutable_input_overwrite_no_return
//       CHECK: util.func public @main$async(
//  CHECK-SAME:     %[[INPUT:.+]]: !hal.buffer_view, %[[MUTABLE_INPUT:.+]]: !hal.buffer_view,
//  CHECK-SAME:     %[[EXTERNAL_BUFFER:.+]]: !hal.buffer,
//  CHECK-SAME:     %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence) -> !hal.buffer_view
//   CHECK-DAG: %[[WAIT_INPUT:.+]] = hal.tensor.import on(#hal.device.promise<@dev_a>) wait(%[[WAIT]]) => %[[INPUT]]
//   CHECK-DAG: %[[TORCH_INPUT:.+]] = torch_c.from_builtin_tensor %[[WAIT_INPUT]]
//   CHECK-DAG: %[[WAIT_MUTABLE_INPUT:.+]] = hal.tensor.import on(#hal.device.promise<@dev_b>) wait(%[[WAIT]]) => %[[MUTABLE_INPUT]]
//   CHECK-DAG: %[[TORCH_MUTABLE_INPUT:.+]] = torch_c.from_builtin_tensor %[[WAIT_MUTABLE_INPUT]]
//   CHECK-DAG: %[[TORCH_RESULT0:.+]] = torch.operator "other_calc"(%[[TORCH_INPUT]])
//   CHECK-DAG: %[[TORCH_RESULT1:.+]] = torch.operator "mutate_inplace"(%[[TORCH_MUTABLE_INPUT]])
//   CHECK-DAG: %[[TENSOR_RESULT0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-DAG: %[[TENSOR_RESULT1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//       CHECK: %[[ALIAS:.+]] = hal.tensor.alias on(#hal.device.promise<@dev_b>) wait(%[[WAIT]]) => %[[TENSOR_RESULT1]] : tensor<5x4xf32> to %[[MUTABLE_INPUT]] : !hal.buffer_view
//   CHECK-DAG: %[[TRANSIENTS_ALIAS:.+]] = hal.tensor.transients %[[ALIAS]] : tensor<5x4xf32> from %[[EXTERNAL_BUFFER]] : !hal.buffer
//   CHECK-DAG: %[[TRANSIENTS_RESULT0:.+]] = hal.tensor.transients %[[TENSOR_RESULT0]] : tensor<4x5xi32> from %[[EXTERNAL_BUFFER]] : !hal.buffer
//       CHECK: %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[TRANSIENTS_ALIAS]], %[[TRANSIENTS_RESULT0]] : tensor<5x4xf32>, tensor<4x5xi32>) => %[[SIGNAL]] : !hal.fence
//   CHECK-DAG: %[[EXPORT_RESULT0:.+]] = hal.tensor.export on(#hal.device.promise<@dev_b>) %[[BARRIER_RESULTS]]#0
//   CHECK-DAG: %[[EXPORT_RESULT1:.+]] = hal.tensor.export on(#hal.device.promise<@dev_a>) %[[BARRIER_RESULTS]]#1
//       CHECK: util.return %[[EXPORT_RESULT1]]
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
