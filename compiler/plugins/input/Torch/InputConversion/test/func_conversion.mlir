// RUN: iree-opt --pass-pipeline="builtin.module(torch-iree-func-conversion)" --allow-unregistered-dialect --split-input-file %s | FileCheck %s --check-prefix=CHECK-ASYNC
// RUN: iree-opt --pass-pipeline="builtin.module(torch-iree-func-conversion{emit-async-entry-points=false})" --allow-unregistered-dialect --split-input-file %s | FileCheck %s --check-prefix=CHECK-SYNC

// Canonical test of the immutable input->compute->return case. This is
// exhaustively verified for both the async and sync wrapper function.
// There shouldn't be much need to further verify the sync wrapper function.
// CHECK-ASYNC-LABEL: @immutable_import_export
//       CHECK-ASYNC: util.func public @main$async(
//  CHECK-ASYNC-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view,
//  CHECK-ASYNC-SAME:     %arg2: !hal.fence, %arg3: !hal.fence) ->
//  CHECK-ASYNC-SAME:     (!hal.buffer_view, !hal.buffer_view)
//  CHECK-ASYNC-SAME:     iree.abi.model = "coarse-fences"
//  CHECK-ASYNC-SAME:     iree.abi.stub
//   CHECK-ASYNC-DAG:   %[[WAIT_ARG0:.+]] = hal.tensor.import wait(%arg2) => %arg0 : !hal.buffer_view -> tensor<4x5xi32>
//   CHECK-ASYNC-DAG:   %[[WAIT_ARG1:.+]] = hal.tensor.import wait(%arg2) => %arg1 : !hal.buffer_view -> tensor<5x4xf32>
//   CHECK-ASYNC-DAG:   %[[TORCH_ARG0:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG0]] : tensor<4x5xi32> -> !torch.vtensor<[4,5],si32>
//   CHECK-ASYNC-DAG:   %[[TORCH_ARG1:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG1]] : tensor<5x4xf32> -> !torch.vtensor<[5,4],f32>
//   CHECK-ASYNC-DAG:   %[[TORCH_RESULT0:.+]] = torch.operator "foobar0"(%[[TORCH_ARG0]])
//   CHECK-ASYNC-DAG:   %[[TORCH_RESULT1:.+]] = torch.operator "foobar1"(%[[TORCH_ARG1]])
//   CHECK-ASYNC-DAG:   %[[TENSOR_RESULT0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-ASYNC-DAG:   %[[TENSOR_RESULT1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//       CHECK-ASYNC:   %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[TENSOR_RESULT0]], %[[TENSOR_RESULT1]] : tensor<4x5xi32>, tensor<5x4xf32>) => %arg3
//   CHECK-ASYNC-DAG:   %[[FUNC_RESULT0:.+]] = hal.tensor.export %[[BARRIER_RESULTS]]#0
//   CHECK-ASYNC-DAG:   %[[FUNC_RESULT1:.+]] = hal.tensor.export %[[BARRIER_RESULTS]]#1
//       CHECK-ASYNC:   util.return %[[FUNC_RESULT0]], %[[FUNC_RESULT1]]
//
//       CHECK-ASYNC: util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view)
//  CHECK-ASYNC-SAME:     -> (!hal.buffer_view, !hal.buffer_view)
//  CHECK-ASYNC-SAME:     iree.abi.stub
//   CHECK-ASYNC-DAG:   %[[CONSTANT0:.+]] = arith.constant 0 : index
//   CHECK-ASYNC-DAG:   %[[CONSTANT1:.+]] = arith.constant -1 : i32
//   CHECK-ASYNC-DAG:   %[[DEVICE0:.+]] = hal.devices.get %[[CONSTANT0]] : !hal.device
//   CHECK-ASYNC-DAG:   %[[NULL_FENCE:.+]] = util.null : !hal.fence
//       CHECK-ASYNC:   %[[NEW_FENCE:.+]] = hal.fence.create device(%[[DEVICE0]] : !hal.device) flags("None")
//       CHECK-ASYNC:   %[[CALL_RESULTS:.+]]:2 = util.call @main$async(%arg0, %arg1, %[[NULL_FENCE]], %[[NEW_FENCE]])
//       CHECK-ASYNC:   %[[AWAIT_STATUS:.+]] = hal.fence.await until([%[[NEW_FENCE]]]) timeout_millis(%[[CONSTANT1]])
//       CHECK-ASYNC:   util.return %[[CALL_RESULTS]]#0, %[[CALL_RESULTS]]#1 : !hal.buffer_view, !hal.buffer_view

// CHECK-SYNC-LABEL: @immutable_import_export
//       CHECK-SYNC: util.func public @main(
//  CHECK-SYNC-SAME:     %arg0: tensor<4x5xi32>, %arg1: tensor<5x4xf32>) ->
//  CHECK-SYNC-SAME:     (tensor<4x5xi32>, tensor<5x4xf32>)
//   CHECK-SYNC-DAG:   %[[TORCH_ARG0:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<4x5xi32> -> !torch.vtensor<[4,5],si32>
//   CHECK-SYNC-DAG:   %[[TORCH_ARG1:.+]] = torch_c.from_builtin_tensor %arg1 : tensor<5x4xf32> -> !torch.vtensor<[5,4],f32>
//   CHECK-SYNC-DAG:   %[[TORCH_RESULT0:.+]] = torch.operator "foobar0"(%[[TORCH_ARG0]])
//   CHECK-SYNC-DAG:   %[[TORCH_RESULT1:.+]] = torch.operator "foobar1"(%[[TORCH_ARG1]])
//   CHECK-SYNC-DAG:   %[[TENSOR_RESULT0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-SYNC-DAG:   %[[TENSOR_RESULT1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//       CHECK-SYNC:   util.return %[[TENSOR_RESULT0]], %[[TENSOR_RESULT1]]
//   CHECK-SYNC-NOT:   hal.tensor
//   CHECK-SYNC-NOT:   hal.fence
//   CHECK-SYNC-NOT:   @main$async
builtin.module @immutable_import_export {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>, %arg1: !torch.vtensor<[5,4],f32>)
    -> (!torch.vtensor<[4,5],si32>, !torch.vtensor<[5,4],f32>) {
  %0 = torch.operator "foobar0"(%arg0) : (!torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>
  %1 = torch.operator "foobar1"(%arg1) : (!torch.vtensor<[5,4],f32>) -> !torch.vtensor<[5,4],f32>
  return %0, %1 : !torch.vtensor<[4,5],si32>, !torch.vtensor<[5,4],f32>
}
}

// -----
// CHECK-ASYNC-LABEL: @return_immutable_arg
// CHECK-ASYNC: util.func public @main$async
// CHECK-ASYNC: hal.fence.signal<%arg2 : !hal.fence>
// CHECK-ASYNC: util.return %arg0

// CHECK-SYNC-LABEL: @return_immutable_arg
// CHECK-SYNC: util.func public @main(
// CHECK-SYNC-SAME:     %arg0: tensor<4x5xi32>) -> tensor<4x5xi32>
// CHECK-SYNC: util.return %arg0
// CHECK-SYNC-NOT: hal.fence
// CHECK-SYNC-NOT: @main$async
builtin.module @return_immutable_arg {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>  {
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-ASYNC-LABEL: @retained_attribute_reflection
//      CHECK-ASYNC: util.func public @main$async(
// CHECK-ASYNC-SAME:   iree.reflection = {some.attr = 4 : index}
//      CHECK-ASYNC: util.func public @main(
// CHECK-ASYNC-SAME:   iree.reflection = {some.attr = 4 : index}

// CHECK-SYNC-LABEL: @retained_attribute_reflection
//      CHECK-SYNC: util.func public @main(
// CHECK-SYNC-SAME:   iree.reflection = {some.attr = 4 : index}
// CHECK-SYNC-NOT: @main$async
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
// CHECK-ASYNC-LABEL: @tied_operand
// CHECK-ASYNC: util.func public @main$async(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> %arg0
// CHECK-ASYNC: util.func public @main(%arg0: !hal.buffer_view) -> !hal.buffer_view
// CHECK-ASYNC: = util.call @main$async{{.*}} -> %arg0

// CHECK-SYNC-LABEL: @tied_operand
// CHECK-SYNC: util.func public @main(%arg0: tensor<4x5xi32>) -> %arg0
// CHECK-SYNC: util.return %arg0
// CHECK-SYNC-NOT: @main$async
builtin.module @tied_operand {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) ->
  (!torch.vtensor<[4,5],si32> {iree.abi.tied = 0})
{
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// Verify that dynamic dimensions verify.
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
// CHECK-ASYNC-LABEL: @torch_bool_return
// CHECK-ASYNC: torch_c.to_i1
// CHECK-ASYNC: util.return {{.*}} : i1

// CHECK-SYNC-LABEL: @torch_bool_return
// CHECK-SYNC: torch_c.to_i1
// CHECK-SYNC: util.return {{.*}} : i1
// CHECK-SYNC-NOT: hal.fence
module @torch_bool_return {
  func.func @main() -> !torch.bool {
    %0 = torch.operator "some.primitive"() : () -> !torch.bool
    return %0 : !torch.bool
  }
}

// -----
// CHECK-ASYNC-LABEL: @torch_int_return
// CHECK-ASYNC: torch_c.to_i64
// CHECK-ASYNC: util.return {{.*}} : i64

// CHECK-SYNC-LABEL: @torch_int_return
// CHECK-SYNC: torch_c.to_i64
// CHECK-SYNC: util.return {{.*}} : i64
// CHECK-SYNC-NOT: hal.fence
module @torch_int_return {
  func.func @main() -> !torch.int {
    %0 = torch.operator "some.primitive"() : () -> !torch.int
    return %0 : !torch.int
  }
}

// -----
// CHECK-LABEL: @torch_float_return
// CHECK: torch_c.to_f64
// CHECK: util.return {{.*}} : f64
module @torch_float_return {
  func.func @main() -> !torch.float {
    %0 = torch.operator "some.primitive"() : () -> !torch.float
    return %0 : !torch.float
  }
}

// -----
// CHECK-LABEL: @torch_generator_return
// CHECK: torch_c.generator_to_i64
// CHECK: util.return {{.*}} : i64
module @torch_generator_return {
  func.func @main() -> !torch.Generator {
    %0 = torch.operator "some.primitive"() : () -> !torch.Generator
    return %0 : !torch.Generator
  }
}

// -----
// CHECK-ASYNC-LABEL: @torch_bool_arg
// CHECK-ASYNC: torch_c.from_i1 %arg0

// CHECK-SYNC-LABEL: @torch_bool_arg
// CHECK-SYNC: torch_c.from_i1 %arg0
// CHECK-SYNC-NOT: hal.fence
module @torch_bool_arg {
  func.func @main(%arg0 : !torch.bool) -> (!torch.vtensor<[1],f32>) {
    %0 = torch.operator "some.primitive"(%arg0) : (!torch.bool) ->  (!torch.vtensor<[1],f32>)
    return %0 : !torch.vtensor<[1],f32>
  }
}

// -----
// CHECK-ASYNC-LABEL: @torch_int_arg
// CHECK-ASYNC: torch_c.from_i64 %arg0

// CHECK-SYNC-LABEL: @torch_int_arg
// CHECK-SYNC: torch_c.from_i64 %arg0
// CHECK-SYNC-NOT: hal.fence
module @torch_int_arg {
  func.func @main(%arg0 : !torch.int) -> (!torch.vtensor<[1],f32>) {
    %0 = torch.operator "some.primitive"(%arg0) : (!torch.int) ->  (!torch.vtensor<[1],f32>)
    return %0 : !torch.vtensor<[1],f32>
  }
}

// -----
// CHECK-LABEL: @torch_float_arg
// CHECK: torch_c.from_f64 %arg0
module @torch_float_arg {
  func.func @main(%arg0 : !torch.float) -> (!torch.vtensor<[1],f32>) {
    %0 = torch.operator "some.primitive"(%arg0) : (!torch.float) ->  (!torch.vtensor<[1],f32>)
    return %0 : !torch.vtensor<[1],f32>
  }
}

// -----
// CHECK-LABEL: @builtin_index_arg
module @builtin_index_arg {
  func.func @main(%arg0 : index) -> (!torch.vtensor<[1],f32>) {
    %0 = "torch_test.operator"(%arg0) : (index) ->  (!torch.vtensor<[1],f32>)
    return %0 : !torch.vtensor<[1],f32>
  }
}

// -----
// CHECK-LABEL: @builtin_int_arg
module @builtin_int_arg {
  func.func @main(%arg0 : i32) -> (!torch.vtensor<[1],f32>) {
    %0 = "torch_test.operator"(%arg0) : (i32) ->  (!torch.vtensor<[1],f32>)
    return %0 : !torch.vtensor<[1],f32>
  }
}

// -----
// CHECK-LABEL: @builtin_float_arg
module @builtin_float_arg {
  func.func @main(%arg0 : f32) -> (!torch.vtensor<[1],f32>) {
    %0 = "torch_test.operator"(%arg0) : (f32) ->  (!torch.vtensor<[1],f32>)
    return %0 : !torch.vtensor<[1],f32>
  }
}

// -----
// CHECK-LABEL: @builtin_index_return
module @builtin_index_return {
  func.func @main() -> index {
    %0 = "torch_test.operator"() : () -> index
    return %0 : index
  }
}

// -----
// CHECK-LABEL: @builtin_int_return
module @builtin_int_return {
  func.func @main() -> i32 {
    %0 = "torch_test.operator"() : () -> i32
    return %0 : i32
  }
}

// -----
// CHECK-LABEL: @builtin_float_return
module @builtin_float_return {
  func.func @main() -> f32 {
    %0 = "torch_test.operator"() : () -> f32
    return %0 : f32
  }
}
