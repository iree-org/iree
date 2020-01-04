// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @scalar
func @scalar() -> tensor<f32> {
  %input1 = iree.unfoldable_constant dense<16.0> : tensor<f32>
  %input2 = iree.unfoldable_constant dense<7.0> : tensor<f32>
  %result = "xla_hlo.remainder"(%input1, %input2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=2

// CHECK-LABEL: EXEC @tensor
func @tensor() -> tensor<3xf32> {
  %input1 = iree.unfoldable_constant dense<[16.0, 17.0, 18.0]> : tensor<3xf32>
  %input2 = iree.unfoldable_constant dense<[7.0, 8.0, 9.0]> : tensor<3xf32>
  %result = "xla_hlo.remainder"(%input1, %input2) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  return %result : tensor<3xf32>
}
// CHECK: f32=2 1 0

// CHECK-LABEL: EXEC @negative_den
func @negative_den() -> tensor<f32> {
  %input1 = iree.unfoldable_constant dense<16.0> : tensor<f32>
  %input2 = iree.unfoldable_constant dense<-7.0> : tensor<f32>
  %result = "xla_hlo.remainder"(%input1, %input2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=2

// CHECK-LABEL: EXEC @negative_num
func @negative_num() -> tensor<f32> {
  %input1 = iree.unfoldable_constant dense<-16.0> : tensor<f32>
  %input2 = iree.unfoldable_constant dense<7.0> : tensor<f32>
  %result = "xla_hlo.remainder"(%input1, %input2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=-2

// CHECK-LABEL: EXEC @scalar_int
func @scalar_int() -> tensor<i32> {
  %input1 = iree.unfoldable_constant dense<16> : tensor<i32>
  %input2 = iree.unfoldable_constant dense<7> : tensor<i32>
  %result = "xla_hlo.remainder"(%input1, %input2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}
// CHECK: i32=2

// CHECK-LABEL: EXEC @tensor_int
func @tensor_int() -> tensor<3xi32> {
  %input1 = iree.unfoldable_constant dense<[16, 17, 18]> : tensor<3xi32>
  %input2 = iree.unfoldable_constant dense<[7, 8, 9]> : tensor<3xi32>
  %result = "xla_hlo.remainder"(%input1, %input2) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  return %result : tensor<3xi32>
}
// CHECK: i32=2 1 0

// CHECK-LABEL: EXEC @negative_den_int
func @negative_den_int() -> tensor<i32> {
  %input1 = iree.unfoldable_constant dense<16> : tensor<i32>
  %input2 = iree.unfoldable_constant dense<-7> : tensor<i32>
  %result = "xla_hlo.remainder"(%input1, %input2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}
// CHECK: i32=2

// CHECK-LABEL: EXEC @negative_num_int
func @negative_num_int() -> tensor<i32> {
  %input1 = iree.unfoldable_constant dense<-16> : tensor<i32>
  %input2 = iree.unfoldable_constant dense<7> : tensor<i32>
  %result = "xla_hlo.remainder"(%input1, %input2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}
// CHECK: i32=-2
