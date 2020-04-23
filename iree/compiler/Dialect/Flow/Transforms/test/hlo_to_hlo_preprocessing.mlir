// RUN: iree-opt -split-input-file -verify-diagnostics -iree-flow-hlo-to-hlo-preprocessing %s | IreeFileCheck %s

// CHECK-LABEL: @batch_norm_inference
// CHECK-SAME: %[[X:[^:[:space:]]+]]
// CHECK-SAME: %[[SCALE:[^:[:space:]]+]]
// CHECK-SAME: %[[OFFSET:[^:[:space:]]+]]
// CHECK-SAME: %[[MEAN:[^:[:space:]]+]]
// CHECK-SAME: %[[VARIANCE:[^:[:space:]]+]]
func @batch_norm_inference(
    %x: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>,
    %mean: tensor<256xf32>, %variance: tensor<256xf32>)
    -> (tensor<4x256xf32>) {
  // CHECK-DAG: %[[EPS:.+]] = xla_hlo.constant dense<1.001000e-05> : tensor<f32>
  // CHECK-DAG: %[[EPS_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[EPS]]) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = xla_hlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = "xla_hlo.sqrt"(%[[VARIANCE_EPS]]) : (tensor<256xf32>) -> tensor<256xf32>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[STDDEV]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[SCALE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[OFFSET]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[MEAN_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[MEAN]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_CENTER:.+]] = xla_hlo.subtract %[[X]], %[[MEAN_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[X_SCALED:.+]] = xla_hlo.multiply %[[X_CENTER]], %[[SCALE_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[X_NORMED:.+]] = xla_hlo.divide %[[X_SCALED]], %[[STDDEV_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[RESULT:.+]] = xla_hlo.add %[[X_NORMED]], %[[OFFSET_BCAST]] : tensor<4x256xf32>
  %0 = "xla_hlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: return %[[RESULT]]
  return %0 : tensor<4x256xf32>
}

// -----

// CHECK: @depth_conv(%[[ARG0:.*]]: tensor<2x4x5x2xf32>, %[[ARG1:.*]]: tensor<2x2x2x3xf32>)
func @depth_conv(%arg0: tensor<2x4x5x2xf32>, %arg1: tensor<2x2x2x3xf32>) -> tensor<2x3x4x6xf32> {
    // CHECK-NOT: xla_hlo.reshape
    //CHECK: "xla_hlo.convolution"(%[[ARG0]], %[[ARG1]])
    %0 = "xla_hlo.reshape"(%arg1) : (tensor<2x2x2x3xf32>) -> tensor<2x2x1x6xf32>
    %1 = "xla_hlo.convolution"(%arg0, %0) {
      batch_group_count = 1 : i64,
      dimension_numbers = {
        input_batch_dimension = 0 : i64,
        input_feature_dimension = 3 : i64,
        input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
        kernel_input_feature_dimension = 2 : i64,
        kernel_output_feature_dimension = 3 : i64,
        kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
        output_batch_dimension = 0 : i64,
        output_feature_dimension = 3 : i64,
        output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
     feature_group_count = 2 : i64,
     padding = dense<0> : tensor<2x2xi64>,
     rhs_dilation = dense<1> : tensor<2xi64>,
     window_strides = dense<1> : tensor<2xi64>} : (tensor<2x4x5x2xf32>, tensor<2x2x1x6xf32>) -> tensor<2x3x4x6xf32>
    return %1 : tensor<2x3x4x6xf32>
}

// -----

// CHECK-LABEL: @reduce_window
func @reduce_window(%input: tensor<1x16x16x64xf32>) -> tensor<1x8x8x64xf32> {
  // CHECK: %[[INITVAL:.*]] = xla_hlo.constant dense<0xFF800000> : tensor<f32>
  %initval = xla_hlo.constant dense<0xFF800000> : tensor<f32>
  //      CHECK: %[[PAD:.*]] = "xla_hlo.pad"(%{{.*}}, %[[INITVAL]])
  // CHECK-SAME: edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>
  // CHECK-SAME: edge_padding_low = dense<[0, 1, 1, 0]> : tensor<4xi64>
  //      CHECK: "xla_hlo.reduce_window"(%[[PAD]], %[[INITVAL]])
  //  CHECK-NOT: padding
  %0 = "xla_hlo.reduce_window"(%input, %initval) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):   // no predecessors
    %3 = xla_hlo.maximum %arg1, %arg2 : tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>
  } : (tensor<1x16x16x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
  return %0 : tensor<1x8x8x64xf32>
}
