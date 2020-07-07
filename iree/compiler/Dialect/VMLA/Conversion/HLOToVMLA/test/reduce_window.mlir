// RUN: iree-opt -split-input-file -iree-vmla-conversion -cse %s | IreeFileCheck %s

// CHECK-LABEL: @pooling_max
func @pooling_max(%arg0: tensor<1x4x6x1xf32>) -> tensor<1x2x2x1xf32>
    attributes { sym_visibility = "private" } {
  // CHECK: vmla.pooling.max
  %cst = constant dense<0.000000e+00> : tensor<f32>
  %0 = "mhlo.reduce_window"(%arg0, %cst) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %1 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 2, 3, 1]> : tensor<4xi64>,
      window_strides = dense<1> : tensor<4xi64>
  } : (tensor<1x4x6x1xf32>, tensor<f32>) -> tensor<1x2x2x1xf32>
  return %0 : tensor<1x2x2x1xf32>
}

// -----

// CHECK-LABEL: @pooling_min
func @pooling_min(%arg0: tensor<1x4x6x1xi32>) -> tensor<1x2x2x1xi32>
    attributes { sym_visibility = "private" } {
  // CHECK: vmla.pooling.min
  %cst = constant dense<0> : tensor<i32>
  %0 = "mhlo.reduce_window"(%arg0, %cst) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
    %1 = mhlo.minimum %arg1, %arg2 : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {window_dimensions = dense<[1, 2, 3, 1]> : tensor<4xi64>,
      window_strides = dense<1> : tensor<4xi64>
  } : (tensor<1x4x6x1xi32>, tensor<i32>) -> tensor<1x2x2x1xi32>
  return %0 : tensor<1x2x2x1xi32>
}

// -----

// CHECK-LABEL: @pooling_sum
func @pooling_sum(%arg0: tensor<4x6xf32>) -> tensor<3x4xf32> attributes
    { sym_visibility = "private" } {
  // CHECK: vmla.pooling.sum
  %cst = constant dense<0.000000e+00> : tensor<f32>
  %0 = "mhlo.reduce_window"(%arg0, %cst) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %1 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[2, 3]> : tensor<2xi64>,
      window_strides = dense<1> : tensor<2xi64>,
      padding = dense<[[1, 0], [2, 0]]> : tensor<2x2xi64>
  } : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
