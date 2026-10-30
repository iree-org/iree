// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --compile-to=flow %s | FileCheck %s

// CHECK-LABEL: util.func public @own_shape
// CHECK-NOT: flow.dispatch
// CHECK: util.return
func.func @own_shape(%arg0: tensor<?x5xf32>) -> tensor<?x5xf32> {
  %0 = shape.shape_of %arg0 : tensor<?x5xf32> -> tensor<2xindex>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [0, 1] : (tensor<?x5xf32>, tensor<2xindex>) -> tensor<?x5xf32>
  return %1 : tensor<?x5xf32>
}

// CHECK-LABEL: util.func public @reshape_shape
// CHECK-NOT: flow.dispatch
// CHECK: flow.tensor.reshape
// CHECK-NOT: flow.dispatch
// CHECK: util.return
func.func @reshape_shape(%arg0: tensor<?xf32>, %arg1: tensor<2xi64>) -> tensor<?x?xf32> {
  %0 = stablehlo.dynamic_reshape %arg0, %arg1 : (tensor<?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  %1 = stablehlo.dynamic_broadcast_in_dim %0, %arg1, dims = [0, 1] : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
