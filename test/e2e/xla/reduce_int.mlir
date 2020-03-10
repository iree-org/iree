// RUN: iree-run-mlir -iree-hal-target-backends=vmla -iree-flow-experimental-dispatch-reduce %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv  -iree-flow-experimental-dispatch-reduce %s)

// Int sum values from [1, 10]
// CHECK-LABEL: EXEC @reduce_sum_1x10xi32
func @reduce_sum_1x10xi32() -> tensor<1xi32> {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<1x10xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xi32>, tensor<i32>) -> tensor<1xi32>
  return %2 : tensor<1xi32>
}
// CHECK: 1xi32=55

// -----

// Int max values from [1, 10]
// CHECK-LABEL: EXEC @reduce_max_1x10xi32
func @reduce_max_1x10xi32() -> tensor<1xi32> {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<1x10xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.max"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xi32>, tensor<i32>) -> tensor<1xi32>
  return %2 : tensor<1xi32>
}
// CHECK: 1xi32=10

// -----

// Int min values, along multiple dimensions. Expected to just be a reshape in this case.
// CHECK-LABEL: EXEC @reduce_min_5x1x1xi32
func @reduce_min_5x1x1xi32() -> tensor<5xi32> {
  %0 = iree.unfoldable_constant dense<[[[1]],[[2]],[[3]],[[4]],[[5]]]> : tensor<5x1x1xi32>
  %1 = iree.unfoldable_constant dense<999> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.min"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<5x1x1xi32>, tensor<i32>) -> tensor<5xi32>
  return %2 : tensor<5xi32>
}
// CHECK: 5xi32=1 2 3 4 5

// -----

// The following cases match the examples presented at
// https://www.tensorflow.org/xla/operation_semantics#reduce

// CHECK-LABEL: EXEC @reduce_sum_2x3xi32_dim0
func @reduce_sum_2x3xi32_dim0() -> tensor<3xi32> {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3],
                       [4, 5, 6]]> : tensor<2x3xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<2x3xi32>, tensor<i32>) -> tensor<3xi32>
  return %2 : tensor<3xi32>
}
// CHECK: 3xi32=5 7 9

// -----

// CHECK-LABEL: EXEC @reduce_sum_2x3xi32_dim1
func @reduce_sum_2x3xi32_dim1() -> tensor<2xi32> {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3],
                       [4, 5, 6]]> : tensor<2x3xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
  return %2 : tensor<2xi32>
}
// CHECK: 2xi32=6 15

// -----

// CHECK-LABEL: EXEC @reduce_sum_4x2x3xi32_dim0
func @reduce_sum_4x2x3xi32_dim0() -> tensor<2x3xi32> {
  %0 = iree.unfoldable_constant dense<[[[1, 2, 3],
                        [4, 5, 6]],
                       [[1, 2, 3],
                        [4, 5, 6]],
                       [[1, 2, 3],
                        [4, 5, 6]],
                       [[1, 2, 3],
                        [4, 5, 6]]]> : tensor<4x2x3xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<4x2x3xi32>, tensor<i32>) -> tensor<2x3xi32>
  return %2 : tensor<2x3xi32>
}
// CHECK: 2x3xi32=[4 8 12][16 20 24]

// -----

// CHECK-LABEL: EXEC @reduce_sum_4x2x3xi32_dim2
func @reduce_sum_4x2x3xi32_dim2() -> tensor<4x2xi32> {
  %0 = iree.unfoldable_constant dense<[[[1, 2, 3],
                        [4, 5, 6]],
                       [[1, 2, 3],
                        [4, 5, 6]],
                       [[1, 2, 3],
                        [4, 5, 6]],
                       [[1, 2, 3],
                        [4, 5, 6]]]> : tensor<4x2x3xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<[2]> : tensor<1xi64>} : (tensor<4x2x3xi32>, tensor<i32>) -> tensor<4x2xi32>
  return %2 : tensor<4x2xi32>
}
// CHECK: 4x2xi32=[6 15][6 15][6 15][6 15]

// -----

// CHECK-LABEL: EXEC @reduce_sum_4x2x3xi32_dims_0_1
func @reduce_sum_4x2x3xi32_dims_0_1() -> tensor<3xi32> {
  %0 = iree.unfoldable_constant dense<[[[1, 2, 3],
                        [4, 5, 6]],
                       [[1, 2, 3],
                        [4, 5, 6]],
                       [[1, 2, 3],
                        [4, 5, 6]],
                       [[1, 2, 3],
                        [4, 5, 6]]]> : tensor<4x2x3xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x2x3xi32>, tensor<i32>) -> tensor<3xi32>
  return %2 : tensor<3xi32>
}
// CHECK: 3xi32=20 28 36

// -----

// CHECK-LABEL: EXEC @reduce_sum_4x2x3xi32_dims_0_1_2
func @reduce_sum_4x2x3xi32_dims_0_1_2() -> tensor<i32> {
  %0 = iree.unfoldable_constant dense<[[[1, 2, 3],
                        [4, 5, 6]],
                       [[1, 2, 3],
                        [4, 5, 6]],
                       [[1, 2, 3],
                        [4, 5, 6]],
                       [[1, 2, 3],
                        [4, 5, 6]]]> : tensor<4x2x3xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<4x2x3xi32>, tensor<i32>) -> tensor<i32>
  return %2 : tensor<i32>
}
// CHECK: i32=84

