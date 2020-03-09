// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -split-input-file -iree-hal-target-backends=vulkan-spirv -iree-flow-experimental-dispatch-reduce %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @reduce_dim_1
// CHECK: 2xi32=25 50
func @reduce_dim_1() -> tensor<2xi32> {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]> : tensor<2x5xi32>
  %1 = iree.unfoldable_constant dense<10> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0 : tensor<i32>, %arg1 : tensor<i32>):
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x5xi32>, tensor<i32>) -> tensor<2xi32>
  return %2 : tensor<2xi32>
}

// -----

// Constants get folded in which linalg.indexed_generic ops. Check to
// make sure this works as expected.
// CHECK-LABEL: EXEC @reduce_dim_1_const
// CHECK: 2xi32=25 50
func @reduce_dim_1_const() -> tensor<2xi32> {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]> : tensor<2x5xi32>
  %1 = constant dense<10> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0 : tensor<i32>, %arg1 : tensor<i32>):
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x5xi32>, tensor<i32>) -> tensor<2xi32>
  return %2 : tensor<2xi32>
}

// -----

// CHECK-LABEL: EXEC @reduce_dim_0
// CHECK: 1xi32=65
func @reduce_dim_0() -> tensor<1xi32> {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<1x10xi32>
  %1 = iree.unfoldable_constant dense<10> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0 : tensor<i32>, %arg1 : tensor<i32>):
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xi32>, tensor<i32>) -> tensor<1xi32>
  return %2 : tensor<1xi32>
}

// -----

// CHECK-LABEL: EXEC @reduce_to_scalar
// CHECK: i32=65
func @reduce_to_scalar() -> tensor<i32> {
  %0 = iree.unfoldable_constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi32>
  %1 = iree.unfoldable_constant dense<10> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0 : tensor<i32>, %arg1 : tensor<i32>):
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
  return %2 : tensor<i32>
}
