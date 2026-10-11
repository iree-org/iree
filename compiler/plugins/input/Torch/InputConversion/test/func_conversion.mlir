// RUN: iree-opt --pass-pipeline="builtin.module(torch-iree-func-conversion)" --allow-unregistered-dialect --split-input-file %s | FileCheck %s

// Canonical test of the immutable input->compute->return case. This is
// exhaustively verified for both the async and sync wrapper function.
// There shouldn't be much need to further verify the sync wrapper function.
// CHECK-LABEL: @immutable_import_export
//       CHECK: util.func public @main$async(
//  CHECK-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view,
//  CHECK-SAME:     %arg2: !hal.fence, %arg3: !hal.fence) ->
//  CHECK-SAME:     (!hal.buffer_view, !hal.buffer_view)
//  CHECK-SAME:     iree.abi.model = "coarse-fences"
//  CHECK-SAME:     iree.abi.stub
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
//  CHECK-SAME:     -> (!hal.buffer_view, !hal.buffer_view)
//  CHECK-SAME:     iree.abi.stub
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
// CHECK: hal.fence.signal<%arg2 : !hal.fence>
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
//       CHECK: %[[EXPORT_ALIAS1:.+]] = hal.tensor.alias wait(%arg2) => %[[TENSOR_ARG1]] : tensor<5x4xf32> to %arg1 : !hal.buffer_view
//       CHECK: %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[EXPORT_ALIAS1]], %[[TENSOR_ARG0]] : tensor<5x4xf32>, tensor<4x5xi32>) => %arg3 : !hal.fence
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
// This isn't a great program to write but is legal. It verifies that if the
// function returns an intermediate vtensor just before it was noted as mutated
// that we export it properly. This would be a hard program to write in PyTorch
// but possible to end up this way so testing the corner.
// Not a good idea to do but legal. This verifies that if returning a mutated
// tensor's intermediate value, you will get two exports, indicating a copy.
// CHECK-LABEL: @mutable_input_overwrite_return_alias_copies
//       CHECK: %[[ALIASED:.+]] = hal.tensor.alias wait({{.+}}) => %{{.+}} : tensor<5x4xf32> to %arg0 : !hal.buffer_view
//       CHECK: %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[ALIASED]], %{{.*}} : tensor<5x4xf32>, tensor<5x4xf32>)
//   CHECK-DAG: = hal.tensor.export %[[BARRIER_RESULTS]]#0
//   CHECK-DAG: = hal.tensor.export %[[BARRIER_RESULTS]]#1
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
// CHECK-LABEL: @mutable_input_overwrite_no_return
//       CHECK: util.func public @main$async(
//  CHECK-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view,
//  CHECK-SAME:     %arg2: !hal.fence, %arg3: !hal.fence) -> !hal.buffer_view
//   CHECK-DAG: %[[WAIT_ARG0:.+]] = hal.tensor.import on(#hal.device.promise<@dev_a>) wait(%arg2) => %arg0
//   CHECK-DAG: %[[TORCH_ARG0:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG0]]
//   CHECK-DAG: %[[WAIT_ARG1:.+]] = hal.tensor.import on(#hal.device.promise<@dev_b>) wait(%arg2) => %arg1
//   CHECK-DAG: %[[TORCH_ARG1:.+]] = torch_c.from_builtin_tensor %[[WAIT_ARG1]]
//   CHECK-DAG: %[[TORCH_RESULT0:.+]] = torch.operator "other_calc"(%[[TORCH_ARG0]])
//   CHECK-DAG: %[[TORCH_RESULT1:.+]] = torch.operator "mutate_inplace"(%[[TORCH_ARG1]])
//   CHECK-DAG: %[[TENSOR_ARG0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-DAG: %[[TENSOR_ARG1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//       CHECK: %[[EXPORT_ALIAS1:.+]] = hal.tensor.alias on(#hal.device.promise<@dev_b>) wait(%arg2) => %[[TENSOR_ARG1]] : tensor<5x4xf32> to %arg1 : !hal.buffer_view
//       CHECK: %[[BARRIER_RESULTS:.+]]:2 = hal.tensor.barrier join(%[[EXPORT_ALIAS1]], %[[TENSOR_ARG0]] : tensor<5x4xf32>, tensor<4x5xi32>) => %arg3 : !hal.fence
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
// CHECK-LABEL: @torch_bool_return
// CHECK: torch_c.to_i1
// CHECK: util.return {{.*}} : i1
module @torch_bool_return {
  func.func @main() -> !torch.bool {
    %0 = torch.operator "some.primitive"() : () -> !torch.bool
    return %0 : !torch.bool
  }
}

// -----
// CHECK-LABEL: @torch_int_return
// CHECK: torch_c.to_i64
// CHECK: util.return {{.*}} : i64
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
// CHECK-LABEL: @torch_bool_arg
// CHECK: torch_c.from_i1 %arg0
module @torch_bool_arg {
  func.func @main(%arg0 : !torch.bool) -> (!torch.vtensor<[1],f32>) {
    %0 = torch.operator "some.primitive"(%arg0) : (!torch.bool) ->  (!torch.vtensor<[1],f32>)
    return %0 : !torch.vtensor<[1],f32>
  }
}

// -----
// CHECK-LABEL: @torch_int_arg
// CHECK: torch_c.from_i64 %arg0
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

// -----

// CHECK-LABEL: @shape_query_static_only
//   CHECK-NOT: $shape_query
//   CHECK-NOT: iree.abi.output_shape_query
//       CHECK: util.func public @main$async
//       CHECK: util.func public @main(
//   CHECK-NOT: $shape_query
//   CHECK-NOT: iree.abi.output_shape_query
module @shape_query_static_only {
  func.func @main(%arg0: !torch.vtensor<[4,5],f32>)
      -> !torch.vtensor<[4,5],f32> {
    %0 = torch.operator "foobar"(%arg0) : (!torch.vtensor<[4,5],f32>) -> !torch.vtensor<[4,5],f32>
    return %0 : !torch.vtensor<[4,5],f32>
  }
}

// -----

// CHECK-LABEL: @shape_query_dynamic_out
//       CHECK: util.func public @main$shape_query$async(
//  CHECK-SAME:     %[[A0:.*arg0]]: !hal.buffer_view, %[[A1:.*arg1]]: !hal.buffer_view,
//  CHECK-SAME:     %[[A2:.*arg2]]: !hal.fence, %arg3: !hal.fence)
//   CHECK-NOT:   util.return {{.*}} : i64
//       CHECK:   %[[SHAPE_IMP:.+]] = hal.tensor.import wait(%[[A2]]) => %[[A1]] : !hal.buffer_view -> tensor<2xi64>
//       CHECK:   %[[SHAPE_VT:.+]] = torch_c.from_builtin_tensor %[[SHAPE_IMP]] : tensor<2xi64> -> !torch.vtensor<[2],si64>
//       CHECK:   %[[SHAPE_BT:.+]] = torch_c.to_builtin_tensor %[[SHAPE_VT]]
//       CHECK:   %[[D0:.+]] = tensor.dim %{{.+}}, %c0 : tensor<?x16xf32>
//       CHECK:   %[[D0_I64:.+]] = arith.index_cast %[[D0]] : index to i64
//       CHECK:   %[[INS:.+]] = tensor.insert %[[D0_I64]] into %[[SHAPE_BT]][%c0] : tensor<2xi64>
//       CHECK:   hal.tensor.alias wait(%[[A2]]) => %[[INS]] : tensor<2xi64> to %[[A1]] : !hal.buffer_view
//       CHECK: util.func public @main$shape_query(
//  CHECK-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view)
//   CHECK-NOT: -> i64
//       CHECK: util.func public @main$async(
//  CHECK-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view,
//  CHECK-SAME:     %arg2: !hal.fence, %arg3: !hal.fence) -> !hal.buffer_view
//  CHECK-SAME:     iree.reflection = {iree.abi.output_shape_query = "main$shape_query"}
//       CHECK: util.func public @main(
//  CHECK-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view
//  CHECK-SAME:     iree.reflection = {iree.abi.output_shape_query = "main$shape_query"}
module @shape_query_dynamic_out {
  func.func @main(
    %arg0: !torch.vtensor<[?,16],f32>,
    %store0: !torch.tensor<[?,16],f32>
  ) -> !torch.vtensor<[?,16],f32> {
    %0 = torch.operator "foobar"(%arg0)
        : (!torch.vtensor<[?,16],f32>) -> !torch.vtensor<[?,16],f32>
    torch.overwrite.tensor.contents %0 overwrites %store0
        : !torch.vtensor<[?,16],f32>, !torch.tensor<[?,16],f32>
    %1 = torch.copy.to_vtensor %store0 : !torch.vtensor<[?,16],f32>
    return %1 : !torch.vtensor<[?,16],f32>
  }
}

// -----

// CHECK-LABEL: @shape_query_mixed_outputs
//       CHECK: util.func public @main$shape_query$async(
//  CHECK-SAME:     %[[A0:.*arg0]]: !hal.buffer_view, %[[A1:.*arg1]]: !hal.buffer_view,
//  CHECK-SAME:     %[[A2:.*arg2]]: !hal.buffer_view, %[[A3:.*arg3]]: !hal.fence,
//  CHECK-SAME:     %[[A4:.*arg4]]: !hal.fence)
//   CHECK-NOT:   util.return {{.*}} : i64
//
// r0 (fully dynamic) -> shape buffer at arg1, every index written.
//       CHECK:   %[[SBUF0:.+]] = hal.tensor.import wait(%[[A3]]) => %[[A1]] : !hal.buffer_view -> tensor<3xi64>
//       CHECK:   %[[SBT0:.+]] = torch_c.to_builtin_tensor %{{.+}} : !torch.vtensor<[3],si64> -> tensor<3xi64>
//       CHECK:   tensor.dim %{{.+}}, %c0 : tensor<?x?x?xf32>
//       CHECK:   tensor.insert %{{.+}} into %[[SBT0]][%c0] : tensor<3xi64>
//       CHECK:   tensor.dim %{{.+}}, %c1 : tensor<?x?x?xf32>
//       CHECK:   tensor.insert %{{.+}} into %{{.+}}[%c1] : tensor<3xi64>
//       CHECK:   tensor.dim %{{.+}}, %c2 : tensor<?x?x?xf32>
//       CHECK:   %[[INS0_LAST:.+]] = tensor.insert %{{.+}} into %{{.+}}[%c2] : tensor<3xi64>
//
// r1 (fully static) contributes nothing; we proceed straight to r2.
//
// r2 (mixed, middle dim static at 8) -> shape buffer at arg2, only indices 0 and 2 are written.
//       CHECK:   %[[SBUF2:.+]] = hal.tensor.import wait(%[[A3]]) => %[[A2]] : !hal.buffer_view -> tensor<3xi64>
//       CHECK:   %[[SBT2:.+]] = torch_c.to_builtin_tensor %{{.+}} : !torch.vtensor<[3],si64> -> tensor<3xi64>
//       CHECK:   tensor.dim %{{.+}}, %{{c0[_0-9]*}} : tensor<?x8x?xf32>
//       CHECK:   tensor.insert %{{.+}} into %[[SBT2]][%{{c0[_0-9]*}}] : tensor<3xi64>
//   CHECK-NOT:   tensor.insert %{{.+}} into %{{.+}}[%{{c1[_0-9]*}}
//       CHECK:   tensor.dim %{{.+}}, %{{c2[_0-9]*}} : tensor<?x8x?xf32>
//       CHECK:   %[[INS2_LAST:.+]] = tensor.insert %{{.+}} into %{{.+}}[%{{c2[_0-9]*}}] : tensor<3xi64>
//
// Both shape buffers are aliased back to their respective shape-buffer args
// after all per-result inserts.
//       CHECK:   hal.tensor.alias wait(%[[A3]]) => %[[INS0_LAST]] : tensor<3xi64> to %[[A1]] : !hal.buffer_view
//       CHECK:   hal.tensor.alias wait(%[[A3]]) => %[[INS2_LAST]] : tensor<3xi64> to %[[A2]] : !hal.buffer_view
//
//       CHECK: util.func public @main$shape_query(
//  CHECK-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view,
//  CHECK-SAME:     %arg2: !hal.buffer_view)
//   CHECK-NOT: -> (i64
//       CHECK: util.func public @main$async(
//  CHECK-SAME:     iree.reflection = {iree.abi.output_shape_query = "main$shape_query"}
//       CHECK: util.func public @main(
//  CHECK-SAME:     iree.reflection = {iree.abi.output_shape_query = "main$shape_query"}
module @shape_query_mixed_outputs {
  func.func @main(
    %arg0: !torch.vtensor<[?,8,?],f32>,
    %store_dyn: !torch.tensor<[?,?,?],f32>,
    %store_static: !torch.tensor<[3,5],f32>,
    %store_mid: !torch.tensor<[?,8,?],f32>
  ) -> (
    !torch.vtensor<[?,?,?],f32>,
    !torch.vtensor<[3,5],f32>,
    !torch.vtensor<[?,8,?],f32>
  ) {
    %r0 = torch.operator "produce_full_dyn"(%arg0)
        : (!torch.vtensor<[?,8,?],f32>) -> !torch.vtensor<[?,?,?],f32>
    %r1 = torch.operator "produce_static"(%arg0)
        : (!torch.vtensor<[?,8,?],f32>) -> !torch.vtensor<[3,5],f32>
    %r2 = torch.operator "produce_mid"(%arg0)
        : (!torch.vtensor<[?,8,?],f32>) -> !torch.vtensor<[?,8,?],f32>
    torch.overwrite.tensor.contents %r0 overwrites %store_dyn
        : !torch.vtensor<[?,?,?],f32>, !torch.tensor<[?,?,?],f32>
    torch.overwrite.tensor.contents %r1 overwrites %store_static
        : !torch.vtensor<[3,5],f32>, !torch.tensor<[3,5],f32>
    torch.overwrite.tensor.contents %r2 overwrites %store_mid
        : !torch.vtensor<[?,8,?],f32>, !torch.tensor<[?,8,?],f32>
    %d0 = torch.copy.to_vtensor %store_dyn    : !torch.vtensor<[?,?,?],f32>
    %d1 = torch.copy.to_vtensor %store_static : !torch.vtensor<[3,5],f32>
    %d2 = torch.copy.to_vtensor %store_mid    : !torch.vtensor<[?,8,?],f32>
    return %d0, %d1, %d2
        : !torch.vtensor<[?,?,?],f32>,
          !torch.vtensor<[3,5],f32>,
          !torch.vtensor<[?,8,?],f32>
  }
}

// -----

// The storage arg `%store0` is read BEFORE being overwritten so the storage arg
// must be kept in the companion's signature.
//
// CHECK-LABEL: @shape_query_kv_cache
//
// The async companion takes the data input and the kept storage arg, plus
// one shape buffer for the dynamic-shape result.
//       CHECK: util.func public @main$shape_query$async(
//  CHECK-SAME:     %[[A0:.*arg0]]: !hal.buffer_view, %[[A1:.*arg1]]: !hal.buffer_view,
//  CHECK-SAME:     %[[A2:.*arg2]]: !hal.buffer_view, %[[A3:.*arg3]]: !hal.fence,
//  CHECK-SAME:     %[[A4:.*arg4]]: !hal.fence)
//   CHECK-NOT:   util.return {{.*}} : i64
//       CHECK:   hal.tensor.import wait(%[[A3]]) => %[[A1]] : !hal.buffer_view -> tensor<?x4xf32>
//       CHECK:   %[[SBUF:.+]] = hal.tensor.import wait(%[[A3]]) => %[[A2]] : !hal.buffer_view -> tensor<2xi64>
//       CHECK:   tensor.dim %{{.+}}, %c0 : tensor<?x4xf32>
//       CHECK:   %[[INS:.+]] = tensor.insert %{{.+}} into %{{.+}}[%c0] : tensor<2xi64>
//       CHECK:   hal.tensor.alias wait(%[[A3]]) => %[[INS]] : tensor<2xi64> to %[[A2]] : !hal.buffer_view
//
//       CHECK: util.func public @main$shape_query(
//  CHECK-SAME:     %arg0: !hal.buffer_view, %arg1: !hal.buffer_view,
//  CHECK-SAME:     %arg2: !hal.buffer_view)
//   CHECK-NOT: -> i64
//       CHECK: util.func public @main$async(
//  CHECK-SAME:     iree.reflection = {iree.abi.output_shape_query = "main$shape_query"}
//       CHECK: util.func public @main(
//  CHECK-SAME:     iree.reflection = {iree.abi.output_shape_query = "main$shape_query"}
module @shape_query_kv_cache {
  func.func @main(
    %arg0: !torch.vtensor<[?,4],f32>,
    %store0: !torch.tensor<[?,4],f32>
  ) -> !torch.vtensor<[?,4],f32> {
    %0 = torch.copy.to_vtensor %store0 : !torch.vtensor<[?,4],f32>
    %1 = torch.operator "foobar"(%arg0, %0)
        : (!torch.vtensor<[?,4],f32>, !torch.vtensor<[?,4],f32>)
       -> !torch.vtensor<[?,4],f32>
    torch.overwrite.tensor.contents %1 overwrites %store0
        : !torch.vtensor<[?,4],f32>, !torch.tensor<[?,4],f32>
    return %1 : !torch.vtensor<[?,4],f32>
  }
}
