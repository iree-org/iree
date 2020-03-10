// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @tensor
func @tensor() -> tensor<4xi32> {
  %lhs = iree.unfoldable_constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = iree.unfoldable_constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "xla_hlo.min"(%lhs, %rhs) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}
// CHECK: 4xi32=1 2 3 4

// -----

// CHECK-LABEL: EXEC @tensor_odd_dim
func @tensor_odd_dim() -> tensor<3xi32> {
  %lhs = iree.unfoldable_constant dense<[1, 2, 7]> : tensor<3xi32>
  %rhs = iree.unfoldable_constant dense<[5, 2, 3]> : tensor<3xi32>
  %result = "xla_hlo.min"(%lhs, %rhs) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  return %result : tensor<3xi32>
}
// CHECK: 3xi32=1 2 3

// -----

// CHECK-LABEL: EXEC @scalar
func @scalar() -> tensor<i32> {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i32>
  %rhs = iree.unfoldable_constant dense<2> : tensor<i32>
  %result = "xla_hlo.min"(%lhs, %rhs) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}
// CHECK: i32=1

// -----

// CHECK-LABEL: EXEC @negative
func @negative() -> tensor<i32> {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i32>
  %rhs = iree.unfoldable_constant dense<-2> : tensor<i32>
  %result = "xla_hlo.min"(%lhs, %rhs) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}
// CHECK: i32=-2

// -----

// CHECK-LABEL: EXEC @i8
func @i8() -> tensor<i8> {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i8>
  %rhs = iree.unfoldable_constant dense<2> : tensor<i8>
  %result = "xla_hlo.min"(%lhs, %rhs) : (tensor<i8>, tensor<i8>) -> tensor<i8>
  return %result : tensor<i8>
}
// CHECK: i8=1

// -----

// CHECK-LABEL: EXEC @i16
func @i16() -> tensor<i16> {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i16>
  %rhs = iree.unfoldable_constant dense<2> : tensor<i16>
  %result = "xla_hlo.min"(%lhs, %rhs) : (tensor<i16>, tensor<i16>) -> tensor<i16>
  return %result : tensor<i16>
}
// CHECK: i16=1

// -----

// CHECK-LABEL: EXEC @i64
func @i64() -> tensor<i64> {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i64>
  %rhs = iree.unfoldable_constant dense<2> : tensor<i64>
  %result = "xla_hlo.min"(%lhs, %rhs) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  return %result : tensor<i64>
}
// CHECK: i32=1
