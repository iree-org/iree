// RUN: iree-opt --pass-pipeline="builtin.module(torch-iree-sync-func-conversion)" --allow-unregistered-dialect --split-input-file %s | FileCheck %s

// Canonical test of the immutable input->compute->return case. The function
// keeps its original name and a plain builtin tensor ABI: no HAL imports or
// exports, no fences, and no $async variant.
// CHECK-LABEL: @immutable_import_export
//       CHECK: util.func public @main(
//  CHECK-SAME:     %arg0: tensor<4x5xi32>, %arg1: tensor<5x4xf32>) ->
//  CHECK-SAME:     (tensor<4x5xi32>, tensor<5x4xf32>)
//   CHECK-DAG:   %[[TORCH_ARG0:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<4x5xi32> -> !torch.vtensor<[4,5],si32>
//   CHECK-DAG:   %[[TORCH_ARG1:.+]] = torch_c.from_builtin_tensor %arg1 : tensor<5x4xf32> -> !torch.vtensor<[5,4],f32>
//   CHECK-DAG:   %[[TORCH_RESULT0:.+]] = torch.operator "foobar0"(%[[TORCH_ARG0]])
//   CHECK-DAG:   %[[TORCH_RESULT1:.+]] = torch.operator "foobar1"(%[[TORCH_ARG1]])
//   CHECK-DAG:   %[[TENSOR_RESULT0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-DAG:   %[[TENSOR_RESULT1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//       CHECK:   util.return %[[TENSOR_RESULT0]], %[[TENSOR_RESULT1]]
//   CHECK-NOT:   hal.tensor
//   CHECK-NOT:   hal.fence
//   CHECK-NOT:   @main$async
builtin.module @immutable_import_export {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>, %arg1: !torch.vtensor<[5,4],f32>)
    -> (!torch.vtensor<[4,5],si32>, !torch.vtensor<[5,4],f32>) {
  %0 = torch.operator "foobar0"(%arg0) : (!torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>
  %1 = torch.operator "foobar1"(%arg1) : (!torch.vtensor<[5,4],f32>) -> !torch.vtensor<[5,4],f32>
  return %0, %1 : !torch.vtensor<[4,5],si32>, !torch.vtensor<[5,4],f32>
}
}

// -----
// A trivially returned argument needs no conversion ops at all.
// CHECK-LABEL: @return_immutable_arg
// CHECK: util.func public @main(
// CHECK-SAME:     %arg0: tensor<4x5xi32>) -> tensor<4x5xi32>
// CHECK-NOT: torch_c.from_builtin_tensor
// CHECK: util.return %arg0
// CHECK-NOT: hal.fence
// CHECK-NOT: @main$async
builtin.module @return_immutable_arg {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>  {
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-LABEL: @retained_attribute_reflection
//      CHECK: util.func public @main(
// CHECK-SAME:   iree.reflection = {some.attr = 4 : index}
// CHECK-NOT: @main$async
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
//      CHECK: util.func public @main(
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
// CHECK-LABEL: @private_visibility
// CHECK: util.func private @main
builtin.module @private_visibility {
func.func private @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>
{
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-LABEL: @tied_operand
// CHECK: util.func public @main(%arg0: tensor<4x5xi32>) -> %arg0
// CHECK: util.return %arg0
// CHECK-NOT: @main$async
builtin.module @tied_operand {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) ->
  (!torch.vtensor<[4,5],si32> {iree.abi.tied = 0})
{
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// Verify that dynamic dimensions convert.
// CHECK-LABEL: @dynamic_dims
// CHECK: util.func public @main(%arg0: tensor<4x?xi32>) -> tensor<4x?xi32>
builtin.module @dynamic_dims {
func.func @main(%arg0: !torch.vtensor<[4,?],si32>) -> !torch.vtensor<[4,?],si32> {
  %0 = torch.operator "foobar0"(%arg0) : (!torch.vtensor<[4,?],si32>) -> !torch.vtensor<[4,?],si32>
  return %0 : !torch.vtensor<[4,?],si32>
}
}

// -----
// CHECK-LABEL: @torch_bool_return
// CHECK: torch_c.to_i1
// CHECK: util.return {{.*}} : i1
// CHECK-NOT: hal.fence
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
// CHECK-NOT: hal.fence
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
// Builtin scalar arguments and results pass through unchanged.
// CHECK-LABEL: @builtin_scalars
// CHECK: util.func public @main(%arg0: index, %arg1: i32, %arg2: f32) -> i32
module @builtin_scalars {
  func.func @main(%arg0 : index, %arg1 : i32, %arg2 : f32) -> i32 {
    %0 = "torch_test.operator"(%arg0, %arg1, %arg2) : (index, i32, f32) -> i32
    return %0 : i32
  }
}
