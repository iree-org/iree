// RUN: iree-opt -split-input-file -verify-diagnostics -iree-mhlo-to-mhlo-preprocessing %s | IreeFileCheck %s

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
  // CHECK-DAG: %[[EPS_BCAST:.+]] = mhlo.constant dense<1.001000e-05> : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = mhlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = "mhlo.sqrt"(%[[VARIANCE_EPS]]) : (tensor<256xf32>) -> tensor<256xf32>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[STDDEV]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[SCALE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[OFFSET]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[MEAN_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[MEAN]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_CENTER:.+]] = mhlo.subtract %[[X]], %[[MEAN_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[X_SCALED:.+]] = mhlo.multiply %[[X_CENTER]], %[[SCALE_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[X_NORMED:.+]] = mhlo.divide %[[X_SCALED]], %[[STDDEV_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[RESULT:.+]] = mhlo.add %[[X_NORMED]], %[[OFFSET_BCAST]] : tensor<4x256xf32>
  %0 = "mhlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: return %[[RESULT]]
  return %0 : tensor<4x256xf32>
}

// -----

// CHECK: @depth_conv(%[[ARG0:.+]]: tensor<2x4x5x2xf32>, %[[ARG1:.+]]: tensor<2x2x2x3xf32>)
func @depth_conv(%arg0: tensor<2x4x5x2xf32>, %arg1: tensor<2x2x2x3xf32>) -> tensor<2x3x4x6xf32> {
    // CHECK-NOT: mhlo.reshape
    // CHECK: mhlo.convolution(%[[ARG0]], %[[ARG1]])
    %0 = "mhlo.reshape"(%arg1) : (tensor<2x2x2x3xf32>) -> tensor<2x2x1x6xf32>
    %1 = "mhlo.convolution"(%arg0, %0) {
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
  // CHECK: %[[INITVAL:.+]] = mhlo.constant dense<0xFF800000> : tensor<f32>
  %initval = mhlo.constant dense<0xFF800000> : tensor<f32>
  //      CHECK: %[[PAD:.+]] = "mhlo.pad"(%{{.+}}, %[[INITVAL]])
  // CHECK-SAME: edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>
  // CHECK-SAME: edge_padding_low = dense<[0, 1, 1, 0]> : tensor<4xi64>
  //      CHECK: "mhlo.reduce_window"(%[[PAD]], %[[INITVAL]])
  //  CHECK-NOT: padding
  %0 = "mhlo.reduce_window"(%input, %initval) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):   // no predecessors
    %3 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%3) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      window_dilations = dense<1> : tensor<4xi64>,
      base_dilations = dense<1> : tensor<4xi64>,
      padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>
  } : (tensor<1x16x16x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
  return %0 : tensor<1x8x8x64xf32>
}

// -----

// CHECK-LABEL: @reduce_window_variadic
func @reduce_window_variadic(%input0: tensor<1x16x16x64xf32>, %input1: tensor<1x16x16x64xi32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>) {
  // CHECK: %[[INITVAL0:.+]] = mhlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: %[[INITVAL1:.+]] = mhlo.constant dense<3> : tensor<i32>
  %initval0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %initval1 = mhlo.constant dense<3> : tensor<i32>

  //      CHECK: %[[PAD0:.+]] = "mhlo.pad"(%{{.+}}, %[[INITVAL0]])
  // CHECK-SAME: edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>
  // CHECK-SAME: edge_padding_low = dense<[0, 1, 1, 0]> : tensor<4xi64>
  //      CHECK: %[[PAD1:.+]] = "mhlo.pad"(%{{.+}}, %[[INITVAL1]])
  // CHECK-SAME: edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>
  // CHECK-SAME: edge_padding_low = dense<[0, 1, 1, 0]> : tensor<4xi64>
  //      CHECK: "mhlo.reduce_window"(%[[PAD0]], %[[PAD1]], %[[INITVAL0]], %[[INITVAL1]])
  //  CHECK-NOT: padding
  %0:2 = "mhlo.reduce_window"(%input0, %input1, %initval0, %initval1) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<i32>):   // no predecessors
    %3 = mhlo.maximum %arg1, %arg3 : tensor<f32>
    %4 = mhlo.add %arg2, %arg4 : tensor<i32>
    "mhlo.return"(%3, %4) : (tensor<f32>, tensor<i32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>
  } : (tensor<1x16x16x64xf32>, tensor<1x16x16x64xi32>, tensor<f32>, tensor<i32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>)
  return %0#0, %0#1 : tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>
}

// -----

func @log_plus_one(%input: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.add
  // CHECK: mhlo.log
  %0 = "mhlo.log_plus_one"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %0: tensor<4xf32>
}

// -----

func @exponential_minus_one(%input: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.exponential
  // CHECK: mhlo.subtract
  %0 = "mhlo.exponential_minus_one"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %0: tensor<4xf32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_scalar_binary(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<i32>, %[[ARG3:.*]]: tensor<i32>)
func @reorder_broadcast_in_dim_scalar_binary(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>) {
  // CHECK: %[[ADD:.*]] = mhlo.add %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[ADD]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[ATAN2:.*]] = mhlo.atan2 %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[ATAN2]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[DIV:.*]] = mhlo.divide %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[DIV]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[MAX:.*]] = mhlo.maximum %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[MAX]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[MIN:.*]] = mhlo.minimum %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[MIN]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[MUL:.*]] = mhlo.multiply %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[MUL]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[POW:.*]] = mhlo.power %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[POW]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[REM:.*]] = mhlo.remainder %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[REM]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SL:.*]] = mhlo.shift_left %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[SL]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SRA:.*]] = mhlo.shift_right_arithmetic %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[SRA]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SRL:.*]] = mhlo.shift_right_logical %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[SRL]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SUB:.*]] = mhlo.subtract %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[SUB]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[AND:.*]] = mhlo.and %[[ARG2]], %[[ARG3]] : tensor<i32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[AND]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>) -> tensor<1x8x8x64xi32>
  // CHECK: %[[OR:.*]] = mhlo.or %[[ARG2]], %[[ARG3]] : tensor<i32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[OR]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>) -> tensor<1x8x8x64xi32>
  // CHECK: %[[XOR:.*]] = mhlo.xor %[[ARG2]], %[[ARG3]] : tensor<i32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[XOR]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>) -> tensor<1x8x8x64xi32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %2 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i32>) -> tensor<1x8x8x64xi32>
  %3 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i32>) -> tensor<1x8x8x64xi32>
  %4 = "mhlo.add"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %5 = "mhlo.atan2"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %6 = "mhlo.divide"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %7 = "mhlo.maximum"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %8 = "mhlo.minimum"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %9 = "mhlo.multiply"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %10 = "mhlo.power"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %11 = "mhlo.remainder"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %12 = "mhlo.shift_left"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %13 = "mhlo.shift_right_arithmetic"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %14 = "mhlo.shift_right_logical"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %15 = "mhlo.subtract"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %16 = "mhlo.and"(%2, %3) : (tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>) -> tensor<1x8x8x64xi32>
  %17 = "mhlo.or"(%2, %3) : (tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>) -> tensor<1x8x8x64xi32>
  %18 = "mhlo.xor"(%2, %3) : (tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>) -> tensor<1x8x8x64xi32>
  return %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18 : tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_scalar_binary_diff_type(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>) -> tensor<1x8x8x64xcomplex<f32>>
func @reorder_broadcast_in_dim_scalar_binary_diff_type(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<1x8x8x64xcomplex<f32>> {
  // CEHCK: %0 = "mhlo.complex"(%[[ARG0]], %[[ARG1]]) : (tensor<f32>, tensor<f32>) -> tensor<complex<f32>>
  // CHECK: "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<complex<f32>>) -> tensor<1x8x8x64xcomplex<f32>>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %2 = "mhlo.complex"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xcomplex<f32>>
  return %2 : tensor<1x8x8x64xcomplex<f32>>
}

// -----

// CHECK: @reorder_broadcast_in_dim_1d_binary(%[[ARG0:.*]]: tensor<3xf32>, %[[ARG1:.*]]: tensor<3xf32>) -> tensor<4x3xf32>
func @reorder_broadcast_in_dim_1d_binary(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<4x3xf32> {
  // CHECK: %[[ATAN2:.*]] = mhlo.atan2 %[[ARG0]], %[[ARG1]] : tensor<3xf32>
  // CHECK: %[[BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[ATAN2]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  %2 = "mhlo.atan2"(%0, %1) : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  // CHECK: return %[[BCAST]]
  return %2 : tensor<4x3xf32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_2d_binary(%[[ARG0:.*]]: tensor<2x4xi32>, %[[ARG1:.*]]: tensor<2x4xi32>) -> tensor<3x2x4xi32>
func @reorder_broadcast_in_dim_2d_binary(%arg0: tensor<2x4xi32>, %arg1: tensor<2x4xi32>) -> tensor<3x2x4xi32> {
  // CHECK: %[[POWER:.*]] = mhlo.power %[[ARG0]], %[[ARG1]] : tensor<2x4xi32>
  // CHECK: %[[BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[POWER]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  %2 = "mhlo.power"(%0, %1) : (tensor<3x2x4xi32>, tensor<3x2x4xi32>) -> tensor<3x2x4xi32>
  // CHECK: return %[[BCAST]]
  return %2 : tensor<3x2x4xi32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_scalar_unary(%[[ARG0:.*]]: tensor<f32>)
func @reorder_broadcast_in_dim_scalar_unary(%arg0: tensor<f32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) {
  // CHECK: %[[ABS:.*]] = "mhlo.abs"(%[[ARG0]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[ABS]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[CEIL:.*]] = "mhlo.ceil"(%[[ARG0]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[CEIL]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[COSINE:.*]] = "mhlo.cosine"(%[[ARG0]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[COSINE]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[EXP:.*]] = "mhlo.exponential"(%[[ARG0]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[EXP]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[FLOOR:.*]] = "mhlo.floor"(%[[ARG0]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[FLOOR]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[LOG:.*]] = "mhlo.log"(%[[ARG0]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[LOG]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[NEG:.*]] = "mhlo.negate"(%[[ARG0]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[NEG]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[ROUND:.*]] = "mhlo.round_nearest_afz"(%[[ARG0]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[ROUND]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[RSQRT:.*]] = "mhlo.rsqrt"(%[[ARG0]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[RSQRT]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SIGN:.*]] = "mhlo.sign"(%[[ARG0]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[SIGN]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SINE:.*]] = "mhlo.sine"(%[[ARG0]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[SINE]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SQRT:.*]] = "mhlo.sqrt"(%[[ARG0]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[SQRT]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[TANH:.*]] = "mhlo.tanh"(%[[ARG0]]) : (tensor<f32>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[TANH]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %1 = "mhlo.abs"(%0) : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %2 = "mhlo.ceil"(%0) : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %3 = "mhlo.cosine"(%0) : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %4 = "mhlo.exponential"(%0) : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %5 = "mhlo.floor"(%0) : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %6 = "mhlo.log"(%0) : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %7 = "mhlo.negate"(%0) : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %8 = "mhlo.round_nearest_afz"(%0) : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %9 = "mhlo.rsqrt"(%0) : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %10 = "mhlo.sign"(%0) : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %11 = "mhlo.sine"(%0) : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %12 = "mhlo.sqrt"(%0) : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %13 = "mhlo.tanh"(%0) : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  return %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13: tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_1d_unary(%[[ARG0:.*]]: tensor<3xf32>) -> tensor<4x3xf32>
func @reorder_broadcast_in_dim_1d_unary(%arg0: tensor<3xf32>) -> tensor<4x3xf32> {
  // CHECK: %[[COS:.*]] = "mhlo.cosine"(%[[ARG0]]) : (tensor<3xf32>) -> tensor<3xf32>
  // CHECK: %[[BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[COS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  %1 = "mhlo.cosine"(%0) : (tensor<4x3xf32>) -> tensor<4x3xf32>
  // CHECK: return %[[BCAST]]
  return %1 : tensor<4x3xf32>
}

// -----

// CHECK: @reorder_in_dim_2d_unary(%[[ARG0:.*]]: tensor<2x4xf32>) -> tensor<3x2x4xf32>
func @reorder_in_dim_2d_unary(%arg0: tensor<2x4xf32>) -> tensor<3x2x4xf32> {
  // CHECK: %[[LOG:.*]] = "mhlo.log"(%[[ARG0]]) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: %[[BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[LOG]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xf32>) -> tensor<3x2x4xf32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xf32>) -> tensor<3x2x4xf32>
  %1 = "mhlo.log"(%0) : (tensor<3x2x4xf32>) -> tensor<3x2x4xf32>
  // CHECK: return %[[BCAST]]
  return %1 : tensor<3x2x4xf32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_scalar_unary_diff_type(%[[ARG0:.*]]: tensor<complex<f32>>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>)
func @reorder_broadcast_in_dim_scalar_unary_diff_type(%arg0: tensor<complex<f32>>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) {
  // CHECK: %[[REAL:.*]] = "mhlo.real"(%[[ARG0]]) : (tensor<complex<f32>>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[REAL]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[IMAG:.*]] = "mhlo.imag"(%[[ARG0]]) : (tensor<complex<f32>>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[IMAG]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<complex<f32>>) -> tensor<1x8x8x64xcomplex<f32>>
  %1 = "mhlo.real"(%0) : (tensor<1x8x8x64xcomplex<f32>>) -> tensor<1x8x8x64xf32>
  %2 = "mhlo.imag"(%0) : (tensor<1x8x8x64xcomplex<f32>>) -> tensor<1x8x8x64xf32>
  return %1, %2: tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>
}
