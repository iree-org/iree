// RUN: iree-opt --pass-pipeline="builtin.module(torch-iree-func-conversion{externalize-transients=true})" --allow-unregistered-dialect --split-input-file %s | FileCheck %s

// This file contains some tests from the parallel file `func_conversion.mlir`, but is run with `externalize-transients=True`.

// CHECK-LABEL: @immutable_import_export
//       CHECK: util.func public @main$async(
//  CHECK-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view,
//  CHECK-SAME:     %[[TRANSIENT_STORAGE:.+]]: !hal.buffer,
//  CHECK-SAME:     %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence) ->
//  CHECK-SAME:     (!hal.buffer_view, !hal.buffer_view)
//  CHECK-SAME:     iree.abi.model = "coarse-fences"
//  CHECK-SAME:     iree.abi.stub
//   CHECK-DAG:   %[[WAIT_ARG0:.+]] = hal.tensor.import wait(%[[WAIT]]) => %arg0 : !hal.buffer_view -> tensor<4x5xi32>
//   CHECK-DAG:   %[[WAIT_ARG1:.+]] = hal.tensor.import wait(%[[WAIT]]) => %arg1 : !hal.buffer_view -> tensor<5x4xf32>
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
//       CHECK: util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer)
//  CHECK-SAME:     -> (!hal.buffer_view, !hal.buffer_view)
//  CHECK-SAME:     iree.abi.stub
//   CHECK-DAG:   %[[CONSTANT0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[CONSTANT1:.+]] = arith.constant -1 : i32
//   CHECK-DAG:   %[[DEVICE0:.+]] = hal.devices.get %[[CONSTANT0]] : !hal.device
//   CHECK-DAG:   %[[NULL_FENCE:.+]] = util.null : !hal.fence
//       CHECK:   %[[NEW_FENCE:.+]] = hal.fence.create device(%[[DEVICE0]] : !hal.device) flags("None")
//       CHECK:   %[[CALL_RESULTS:.+]]:2 = util.call @main$async(%arg0, %arg1, %arg2, %[[NULL_FENCE]], %[[NEW_FENCE]])
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
// CHECK-LABEL: @return_immutable_arg
// CHECK: util.func public @main$async
// CHECK: hal.fence.signal<%arg3 : !hal.fence>
// CHECK: util.return %arg0
// CHECK-NOT: hal.tensor.transients
builtin.module @return_immutable_arg {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>  {
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-LABEL: @mutable_input_overwrite_no_return
//       CHECK: util.func public @main$async(
//  CHECK-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view,
//  CHECK-SAME:     %arg2: !hal.buffer,
//  CHECK-SAME:     %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view
//   CHECK-DAG: %[[WAIT_ARG0:.+]] = hal.tensor.import wait(%arg3) => %arg0
//   CHECK-DAG: %[[TORCH_ARG0:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG0]]
//   CHECK-DAG: %[[WAIT_ARG1:.+]] = hal.tensor.import wait(%arg3) => %arg1
//   CHECK-DAG: %[[TORCH_ARG1:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG1]]
//   CHECK-DAG: %[[TORCH_RESULT0:.+]] = torch.operator "other_calc"(%[[TORCH_ARG0]])
//   CHECK-DAG: %[[TORCH_RESULT1:.+]] = torch.operator "mutate_inplace"(%[[TORCH_ARG1]])
//   CHECK-DAG: %[[TENSOR_ARG0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-DAG: %[[TENSOR_ARG1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//       CHECK: %[[EXPORT_ALIAS1:.+]] = hal.tensor.alias wait(%arg3) => %[[TENSOR_ARG1]] : tensor<5x4xf32> to %arg1 : !hal.buffer_view
//   CHECK-DAG: %[[TRANSIENT_CALL1:.+]] = hal.tensor.transients %[[EXPORT_ALIAS1]] : tensor<5x4xf32> from %arg2 : !hal.buffer
//   CHECK-DAG: %[[TRANSIENT_CALL0:.+]] = hal.tensor.transients %[[TENSOR_ARG0]] : tensor<4x5xi32> from %arg2 : !hal.buffer
//       CHECK: %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[TRANSIENT_CALL1]], %[[TRANSIENT_CALL0]] : tensor<5x4xf32>, tensor<4x5xi32>) => %arg4 : !hal.fence
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
// CHECK-LABEL: @immutable_import_export
// CHECK: hal.buffer_view.dim<%arg0
// CHECK: hal.buffer_view.dim<%arg1
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
//  CHECK-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view,
//  CHECK-SAME:     %arg2: !hal.buffer,
//  CHECK-SAME:     %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view
//   CHECK-DAG: %[[WAIT_ARG0:.+]] = hal.tensor.import on(#hal.device.promise<@dev_a>) wait(%arg3) => %arg0
//   CHECK-DAG: %[[TORCH_ARG0:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG0]]
//   CHECK-DAG: %[[WAIT_ARG1:.+]] = hal.tensor.import on(#hal.device.promise<@dev_b>) wait(%arg3) => %arg1
//   CHECK-DAG: %[[TORCH_ARG1:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG1]]
//   CHECK-DAG: %[[TORCH_RESULT0:.+]] = torch.operator "other_calc"(%[[TORCH_ARG0]])
//   CHECK-DAG: %[[TORCH_RESULT1:.+]] = torch.operator "mutate_inplace"(%[[TORCH_ARG1]])
//   CHECK-DAG: %[[TENSOR_ARG0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-DAG: %[[TENSOR_ARG1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//       CHECK: %[[EXPORT_ALIAS1:.+]] = hal.tensor.alias on(#hal.device.promise<@dev_b>) wait(%arg3) => %[[TENSOR_ARG1]] : tensor<5x4xf32> to %arg1 : !hal.buffer_view
//   CHECK-DAG: %[[TRANSIENTS_CALL1:.+]] = hal.tensor.transients %[[EXPORT_ALIAS1]] : tensor<5x4xf32> from %arg2 : !hal.buffer
//   CHECK-DAG: %[[TRANSIENTS_CALL0:.+]] = hal.tensor.transients %[[TENSOR_ARG0]] : tensor<4x5xi32> from %arg2 : !hal.buffer
//       CHECK: %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[TRANSIENTS_CALL1]], %[[TRANSIENTS_CALL0]] : tensor<5x4xf32>, tensor<4x5xi32>) => %arg4 : !hal.fence
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
