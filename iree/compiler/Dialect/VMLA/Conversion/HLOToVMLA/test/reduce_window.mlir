// RUN: iree-opt -split-input-file -iree-vmla-conversion -cse %s -verify-diagnostics | IreeFileCheck %s

// CHECK-LABEL: @pooling_max
func private @pooling_max(%arg0: tensor<1x4x6x1xf32>) -> tensor<1x2x2x1xf32> {
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
func private @pooling_min(%arg0: tensor<1x4x6x1xi32>) -> tensor<1x2x2x1xi32> {
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
func private @pooling_sum(%arg0: tensor<4x6xf32>) -> tensor<3x4xf32> {
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

// -----

// CHECK-LABEL: @pooling_sum_min
func private @pooling_sum_min(%arg0: tensor<4x6xf32>) -> (tensor<3x4xf32>, tensor<3x4xf32>) {
  // CHECK: vmla.pooling.sum
  // CHECK: vmla.pooling.min
  %cst = constant dense<0.000000e+00> : tensor<f32>
  %0:2 = "mhlo.reduce_window"(%arg0, %arg0, %cst, %cst) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
    %1 = mhlo.add %arg1, %arg3 : tensor<f32>
    %2 = mhlo.minimum %arg2, %arg4 : tensor<f32>
    "mhlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()
  }) {window_dimensions = dense<[2, 3]> : tensor<2xi64>,
      window_strides = dense<1> : tensor<2xi64>,
      padding = dense<[[1, 0], [2, 0]]> : tensor<2x2xi64>
  } : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<3x4xf32>, tensor<3x4xf32>)
  return %0#0, %0#1 : tensor<3x4xf32>, tensor<3x4xf32>
}

// -----

// Specify the module explicitly to anchor the conversion failure message.
// expected-error@+1 {{conversion to the VMLA dialect failed}}
module {

  func private @pooling_sum_min_fail(%arg0: tensor<4x6xf32>) -> (tensor<3x4xf32>, tensor<3x4xf32>) {
    %cst = constant dense<0.000000e+00> : tensor<f32>
    // expected-remark @+2 {{unsupported builtin reduction operation}}
    // expected-error @+1 {{failed to legalize operation 'mhlo.reduce_window' that was explicitly marked illegal}}
    %0:2 = "mhlo.reduce_window"(%arg0, %arg0, %cst, %cst) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %1 = mhlo.add %arg1, %arg2 : tensor<f32>
      %2 = mhlo.minimum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()
    }) {window_dimensions = dense<[2, 3]> : tensor<2xi64>,
        window_strides = dense<1> : tensor<2xi64>,
        padding = dense<[[1, 0], [2, 0]]> : tensor<2x2xi64>
    } : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<3x4xf32>, tensor<3x4xf32>)
    return %0#0, %0#1 : tensor<3x4xf32>, tensor<3x4xf32>
  }
}
