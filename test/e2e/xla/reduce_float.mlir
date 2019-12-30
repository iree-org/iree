// RUN: iree-run-mlir %s -iree-hal-target-backends=interpreter-bytecode | IreeFileCheck %s
// TODO(b/142903911): figure out swiftshader+asan crash:
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir %s -iree-hal-target-backends=vulkan-spirv --run=false)

// Float sum values from [1.0, 10.0]
// CHECK-LABEL: EXEC @reduce_sum_1x10xf32
func @reduce_sum_1x10xf32() -> tensor<1xf32> {
  %0 = constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]> : tensor<1x10xf32>
  %1 = constant dense<0.0> : tensor<f32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
  return %2 : tensor<1xf32>
}
// CHECK: 1xf32=55

// -----

// Float max values from [1.0, 10.0]
// CHECK-LABEL: EXEC @reduce_max_1x10xf32
func @reduce_max_1x10xf32() -> tensor<1xf32> {
  %0 = constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]> : tensor<1x10xf32>
  %1 = constant dense<0.0> : tensor<f32>
  %2 = "xla_hlo.reduce"(%0, %1)
  ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
      %3 = "xla_hlo.max"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  })
  {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
  return %2 : tensor<1xf32>
}
// CHECK: 1xf32=10

// -----

// Float min values, along multiple dimensions. Expected to just be a reshape in this case.
// CHECK-LABEL: EXEC @reduce_min_5x1x1xf32
func @reduce_min_5x1x1xf32() -> tensor<5xf32> {
  %0 = constant dense<[[[1.0]],[[2.0]],[[3.0]],[[4.0]],[[5.0]]]> : tensor<5x1x1xf32>
  %1 = constant dense<999.0> : tensor<f32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
      %3 = "xla_hlo.min"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<5x1x1xf32>, tensor<f32>) -> tensor<5xf32>
  return %2 : tensor<5xf32>
}
// CHECK: 5xf32=1 2 3 4 5

// -----

// The following cases match the examples presented at
// https://www.tensorflow.org/xla/operation_semantics#reduce

// CHECK-LABEL: EXEC @reduce_sum_2x3xf32_dim0
func @reduce_sum_2x3xf32_dim0() -> tensor<3xf32> {
  %0 = constant dense<[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %1 = constant dense<0.0> : tensor<f32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
  return %2 : tensor<3xf32>
}
// CHECK: 3xf32=5 7 9

// -----

// CHECK-LABEL: EXEC @reduce_sum_2x3xf32_dim1
func @reduce_sum_2x3xf32_dim1() -> tensor<2xf32> {
  %0 = constant dense<[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %1 = constant dense<0.0> : tensor<f32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
  return %2 : tensor<2xf32>
}
// CHECK: 2xf32=6 15

// -----

// CHECK-LABEL: EXEC @reduce_sum_4x2x3xf32_dim0
func @reduce_sum_4x2x3xf32_dim0() -> tensor<2x3xf32> {
  %0 = constant dense<[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = constant dense<0.0> : tensor<f32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return %2 : tensor<2x3xf32>
}
// CHECK: 2x3xf32=[4 8 12][16 20 24]

// -----

// CHECK-LABEL: EXEC @reduce_sum_4x2x3xf32_dim1
func @reduce_sum_4x2x3xf32_dim1() -> tensor<4x3xf32> {
  %0 = constant dense<[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = constant dense<0.0> : tensor<f32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<4x3xf32>
  return %2 : tensor<4x3xf32>
}
// CHECK: 4x3xf32=[5 7 9][5 7 9][5 7 9][5 7 9]

// -----

// CHECK-LABEL: EXEC @reduce_sum_4x2x3xf32_dim2
func @reduce_sum_4x2x3xf32_dim2() -> tensor<4x2xf32> {
  %0 = constant dense<[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = constant dense<0.0> : tensor<f32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<[2]> : tensor<1xi64>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<4x2xf32>
  return %2 : tensor<4x2xf32>
}
// CHECK: 4x2xf32=[6 15][6 15][6 15][6 15]

// -----

// CHECK-LABEL: EXEC @reduce_sum_4x2x3xf32_dims_0_1
func @reduce_sum_4x2x3xf32_dims_0_1() -> tensor<3xf32> {
  %0 = constant dense<[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = constant dense<0.0> : tensor<f32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<3xf32>
  return %2 : tensor<3xf32>
}
// CHECK: 3xf32=20 28 36

// -----

// CHECK-LABEL: EXEC @reduce_sum_4x2x3xf32_dims_0_1_2
func @reduce_sum_4x2x3xf32_dims_0_1_2() -> tensor<f32> {
  %0 = constant dense<[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = constant dense<0.0> : tensor<f32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}
// CHECK: f32=84
