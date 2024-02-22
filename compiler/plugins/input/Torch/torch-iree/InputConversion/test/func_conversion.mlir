// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(torch-iree-func-conversion)" %s | FileCheck %s

// Canonical test of the immutable input->compute->return case. This is
// exhaustively verified for both the async and sync wrapper function.
// There shouldn't be much need to further verify the sync wrapper function.
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

// -----
// Tests the immutable + mutable argument case where the mutable argument is
// overwritten as part of the function cleanup and the argument is not returned.
// This exhaustively verifies the async function.
// Note that the order of the barrier operands and successors is implementation
// dependent, and the current implementation processes mutable before
// immutable.
// CHECK-LABEL: @mutable_input_overwrite_no_return
//       CHECK: util.func public @main$async(
//  CHECK-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view, 
//  CHECK-SAME:     %arg2: !hal.fence, %arg3: !hal.fence) -> !hal.buffer_view
//   CHECK-DAG: %[[WAIT_ARG0:.+]] = hal.tensor.import wait(%arg2) => %arg0
//   CHECK-DAG: %[[TORCH_ARG0:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG0]]
//   CHECK-DAG: %[[WAIT_ARG1:.+]] = hal.tensor.import wait(%arg2) => %arg1
//   CHECK-DAG: %[[TORCH_ARG1:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG1]]
//   CHECK-DAG: %[[TORCH_RESULT0:.+]] = torch.operator "other_calc"(%[[TORCH_ARG0]])
//   CHECK-DAG: %[[TORCH_RESULT1:.+]] = torch.operator "mutate_inplace"(%[[TORCH_ARG1]])
//   CHECK-DAG: %[[TENSOR_ARG0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-DAG: %[[TENSOR_ARG1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//       CHECK: %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[TENSOR_ARG1]], %[[TENSOR_ARG0]] : tensor<5x4xf32>, tensor<4x5xi32>) => %arg3 : !hal.fence
//   CHECK-DAG: %[[EXPORT_RESULT1:.+]] = hal.tensor.export %[[BARRIER_RESULTS]]#0 into(%arg1 : !hal.buffer_view)
//   CHECK-DAG: %[[UNUSED:.+]] = util.optimization_barrier %[[EXPORT_RESULT1]]
//   CHECK-DAG: %[[EXPORT_RESULT0:.+]] = hal.tensor.export %[[BARRIER_RESULTS]]#1 :
//       CHECK: util.return %[[EXPORT_RESULT0]]
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
// CHECK-LABEL: @mutable_input_overwrite_return_alias_copies
//       CHECK: %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%{{.*}}, %{{.*}} : tensor<5x4xf32>, tensor<5x4xf32>) 
//   CHECK-DAG: = hal.tensor.export %[[BARRIER_RESULTS]]#0 into(%arg0 : !hal.buffer_view)
//   CHECK-DAG: = hal.tensor.export %[[BARRIER_RESULTS]]#1 :
builtin.module @mutable_input_overwrite_return_alias_copies {
func.func @main(%arg0: !torch.tensor<[5,4],f32>) -> (!torch.vtensor<[5,4],f32>) {
  %0 = torch.copy.to_vtensor %arg0 : !torch.vtensor<[5,4],f32>
  %1 = torch.operator "mutate_inplace"(%0) : (!torch.vtensor<[5,4],f32>) -> !torch.vtensor<[5,4],f32>
  torch.overwrite.tensor.contents %1 overwrites %arg0 : !torch.vtensor<[5,4],f32>, !torch.tensor<[5,4],f32>
  return %1 : !torch.vtensor<[5,4],f32>
}
}

// -----
// CHECK-LABEL: @retained_attribute_reflection
//      CHECK: util.func public @main$async(
// CHECK-SAME:   iree.reflection = {some.attr = 4 : index}
//      CHECK: util.func public @main(
// CHECK-SAME:   iree.reflection = {some.attr = 4 : index}
builtin.module @retained_attribute_reflection {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32> 
  attributes {
    iree.reflection = {
      some.attr = 4 : index
    }    
  }
{
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-LABEL: @retained_attribute_ignored
//      CHECK: util.func public @main$async(
//  CHECK-NOT: iree.nonretained
builtin.module @retained_attribute_ignored {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32> 
  attributes {
    iree.nonretained = "dummy"
  }
{
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-LABEL: @retained_attribute_noinline
//      CHECK: util.func public @main$async(
// CHECK-SAME:   inlining_policy = #util.inline.never
//      CHECK: util.func public @main(
// CHECK-NOT:    inlining_policy
builtin.module @retained_attribute_noinline {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32> 
  attributes {
    noinline
  }
{
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-LABEL: @private_visibility
// CHECK: util.func private @main$async
// CHECK: util.func private @main
builtin.module @private_visibility {
func.func private @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32> 
{
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-LABEL: @tied_operand
// CHECK: util.func public @main$async(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> %arg0
// CHECK: util.func public @main(%arg0: !hal.buffer_view) -> !hal.buffer_view
// CHECK: = util.call @main$async{{.*}} -> %arg0
builtin.module @tied_operand {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> 
  (!torch.vtensor<[4,5],si32> {iree.abi.tied = 0})
{
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}
