// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(torch-iree-func-conversion)" %s | FileCheck %s

// CHECK-LABEL: @immutable_import_export
//       CHECK: util.func public @main$async(
//  CHECK-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view, 
//  CHECK-SAME:     %arg2: !hal.fence, %arg3: !hal.fence) -> 
//  CHECK-SAME:     (!hal.buffer_view, !hal.buffer_view) 
//  CHECK-SAME:     attributes {iree.abi.model = "coarse-fences", iree.abi.stub} 
//   CHECK-DAG:   %[[WAIT_ARG0:.+]] = hal.tensor.import wait(%arg2) => %arg0 : !hal.buffer_view -> tensor<4x5xi32>
//   CHECK-DAG:   %[[WAIT_ARG1:.+]] = hal.tensor.import wait(%arg2) => %arg1 : !hal.buffer_view -> tensor<5x4xf32>
//   CHECK-DAG:   %[[TORCH_ARG0:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG0]] : tensor<4x5xi32> -> !torch.vtensor<[4,5],si32>
//   CHECK-DAG:   %[[TORCH_ARG1:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG1]] : tensor<5x4xf32> -> !torch.vtensor<[5,4],f32>
//   CHECK-DAG:   %[[TORCH_RESULT0:.+]] = torch.operator "foobar0"(%[[TORCH_ARG0]])
//   CHECK-DAG:   %[[TORCH_RESULT1:.+]] = torch.operator "foobar1"(%[[TORCH_ARG1]])
//   CHECK-DAG:   %[[TENSOR_RESULT0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-DAG:   %[[TENSOR_RESULT1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//       CHECK:   %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[TENSOR_RESULT0]], %[[TENSOR_RESULT1]] : tensor<4x5xi32>, tensor<5x4xf32>) => %arg3
//   CHECK-DAG:   %[[FUNC_RESULT0:.+]] = hal.tensor.export %[[BARRIER_RESULTS]]#0
//   CHECK-DAG:   %[[FUNC_RESULT1:.+]] = hal.tensor.export %[[BARRIER_RESULTS]]#1
//       CHECK:   util.return %[[FUNC_RESULT0]], %[[FUNC_RESULT1]]
//
//       CHECK: util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) 
//  CHECK-SAME:     -> (!hal.buffer_view, !hal.buffer_view) attributes {iree.abi.stub}
//   CHECK-DAG:   %[[CONSTANT0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[CONSTANT1:.+]] = arith.constant -1 : i32
//   CHECK-DAG:   %[[DEVICE0:.+]] = hal.devices.get %[[CONSTANT0]] : !hal.device
//   CHECK-DAG:   %[[NULL_FENCE:.+]] = util.null : !hal.fence
//       CHECK:   %[[NEW_FENCE:.+]] = hal.fence.create device(%[[DEVICE0]] : !hal.device) flags("None")
//       CHECK:   %[[CALL_RESULTS:.+]]:2 = util.call @main$async(%arg0, %arg1, %[[NULL_FENCE]], %[[NEW_FENCE]])
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
// CHECK: hal.tensor.barrier join( : ) => %arg2
// CHECK: util.return %arg0
builtin.module @return_immutable_arg {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>  {
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}
