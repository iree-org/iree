// RUN: iree-opt --split-input-file --verify-diagnostics --iree-mhlo-to-mhlo-preprocessing %s | FileCheck %s

// CHECK-LABEL: @batch_norm_inference
// CHECK-SAME: %[[X:[^:[:space:]]+]]
// CHECK-SAME: %[[SCALE:[^:[:space:]]+]]
// CHECK-SAME: %[[OFFSET:[^:[:space:]]+]]
// CHECK-SAME: %[[MEAN:[^:[:space:]]+]]
// CHECK-SAME: %[[VARIANCE:[^:[:space:]]+]]
func.func @batch_norm_inference(
    %x: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>,
    %mean: tensor<256xf32>, %variance: tensor<256xf32>)
    -> (tensor<4x256xf32>) {
  // CHECK-DAG: %[[EPS_BCAST:.+]] = mhlo.constant dense<1.001000e-05> : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = mhlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = mhlo.sqrt %[[VARIANCE_EPS]] : tensor<256xf32>
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

// CHECK: @reorder_broadcast_in_dim_scalar_binary(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<i32>, %[[ARG3:.*]]: tensor<i32>)
func.func @reorder_broadcast_in_dim_scalar_binary(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>) {
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
  // CHECK: %[[SL:.*]] = mhlo.shift_left %[[ARG2]], %[[ARG3]] : tensor<i32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[SL]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>) -> tensor<1x8x8x64xi32>
  // CHECK: %[[SRA:.*]] = mhlo.shift_right_arithmetic %[[ARG2]], %[[ARG3]] : tensor<i32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[SRA]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>) -> tensor<1x8x8x64xi32>
  // CHECK: %[[SRL:.*]] = mhlo.shift_right_logical %[[ARG2]], %[[ARG3]] : tensor<i32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[SRL]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>) -> tensor<1x8x8x64xi32>
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
  %4 = mhlo.add %0, %1 : tensor<1x8x8x64xf32>
  %5 = mhlo.atan2 %0, %1 : tensor<1x8x8x64xf32>
  %6 = mhlo.divide %0, %1 : tensor<1x8x8x64xf32>
  %7 = mhlo.maximum %0, %1 : tensor<1x8x8x64xf32>
  %8 = mhlo.minimum %0, %1 : tensor<1x8x8x64xf32>
  %9 = mhlo.multiply %0, %1 : tensor<1x8x8x64xf32>
  %10 = mhlo.power %0, %1 : tensor<1x8x8x64xf32>
  %11 = mhlo.remainder %0, %1 : tensor<1x8x8x64xf32>
  %12 = mhlo.shift_left %2, %3 : tensor<1x8x8x64xi32>
  %13 = mhlo.shift_right_arithmetic %2, %3 : tensor<1x8x8x64xi32>
  %14 = mhlo.shift_right_logical %2, %3 : tensor<1x8x8x64xi32>
  %15 = mhlo.subtract %0, %1 : tensor<1x8x8x64xf32>
  %16 = mhlo.and %2, %3 : tensor<1x8x8x64xi32>
  %17 = mhlo.or %2, %3 : tensor<1x8x8x64xi32>
  %18 = mhlo.xor %2, %3 : tensor<1x8x8x64xi32>
  return %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18 : tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_scalar_binary_diff_type(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>) -> tensor<1x8x8x64xcomplex<f32>>
func.func @reorder_broadcast_in_dim_scalar_binary_diff_type(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<1x8x8x64xcomplex<f32>> {
  // CHECK: %0 = mhlo.complex %[[ARG0]], %[[ARG1]] : tensor<complex<f32>>
  // CHECK: "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<complex<f32>>) -> tensor<1x8x8x64xcomplex<f32>>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %2 = "mhlo.complex"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xcomplex<f32>>
  return %2 : tensor<1x8x8x64xcomplex<f32>>
}

// -----

// CHECK: @reorder_broadcast_in_dim_1d_binary(%[[ARG0:.*]]: tensor<3xf32>, %[[ARG1:.*]]: tensor<3xf32>) -> tensor<4x3xf32>
func.func @reorder_broadcast_in_dim_1d_binary(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<4x3xf32> {
  // CHECK: %[[ATAN2:.*]] = mhlo.atan2 %[[ARG0]], %[[ARG1]] : tensor<3xf32>
  // CHECK: %[[BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[ATAN2]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  %2 = mhlo.atan2 %0, %1 : tensor<4x3xf32>
  // CHECK: return %[[BCAST]]
  return %2 : tensor<4x3xf32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_2d_binary(%[[ARG0:.*]]: tensor<2x4xi32>, %[[ARG1:.*]]: tensor<2x4xi32>) -> tensor<3x2x4xi32>
func.func @reorder_broadcast_in_dim_2d_binary(%arg0: tensor<2x4xi32>, %arg1: tensor<2x4xi32>) -> tensor<3x2x4xi32> {
  // CHECK: %[[POWER:.*]] = mhlo.power %[[ARG0]], %[[ARG1]] : tensor<2x4xi32>
  // CHECK: %[[BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[POWER]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  %2 = mhlo.power %0, %1 : tensor<3x2x4xi32>
  // CHECK: return %[[BCAST]]
  return %2 : tensor<3x2x4xi32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_scalar_unary(%[[ARG0:.*]]: tensor<f32>)
func.func @reorder_broadcast_in_dim_scalar_unary(%arg0: tensor<f32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) {
  // CHECK: %[[ABS:.*]] = mhlo.abs %[[ARG0]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[ABS]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[CEIL:.*]] = mhlo.ceil %[[ARG0]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[CEIL]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[COSINE:.*]] = mhlo.cosine %[[ARG0]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[COSINE]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[EXP:.*]] = mhlo.exponential %[[ARG0]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[EXP]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[FLOOR:.*]] = mhlo.floor %[[ARG0]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[FLOOR]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[LOG:.*]] = mhlo.log %[[ARG0]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[LOG]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[NEG:.*]] = mhlo.negate %[[ARG0]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[NEG]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[ROUND:.*]] = mhlo.round_nearest_afz %[[ARG0]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[ROUND]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[RSQRT:.*]] = mhlo.rsqrt %[[ARG0]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[RSQRT]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SIGN:.*]] = mhlo.sign %[[ARG0]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[SIGN]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SINE:.*]] = mhlo.sine %[[ARG0]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[SINE]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SQRT:.*]] = mhlo.sqrt %[[ARG0]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[SQRT]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[TANH:.*]] = mhlo.tanh %[[ARG0]] : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[TANH]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %1 = mhlo.abs %0 : tensor<1x8x8x64xf32>
  %2 = mhlo.ceil %0 : tensor<1x8x8x64xf32>
  %3 = mhlo.cosine %0 : tensor<1x8x8x64xf32>
  %4 = mhlo.exponential %0 : tensor<1x8x8x64xf32>
  %5 = mhlo.floor %0 : tensor<1x8x8x64xf32>
  %6 = mhlo.log %0 : tensor<1x8x8x64xf32>
  %7 = mhlo.negate %0 : tensor<1x8x8x64xf32>
  %8 = mhlo.round_nearest_afz %0 : tensor<1x8x8x64xf32>
  %9 = mhlo.rsqrt %0 : tensor<1x8x8x64xf32>
  %10 = mhlo.sign %0 : tensor<1x8x8x64xf32>
  %11 = mhlo.sine %0 : tensor<1x8x8x64xf32>
  %12 = mhlo.sqrt %0 : tensor<1x8x8x64xf32>
  %13 = mhlo.tanh %0 : tensor<1x8x8x64xf32>
  return %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13: tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_1d_unary(%[[ARG0:.*]]: tensor<3xf32>) -> tensor<4x3xf32>
func.func @reorder_broadcast_in_dim_1d_unary(%arg0: tensor<3xf32>) -> tensor<4x3xf32> {
  // CHECK: %[[COS:.*]] = mhlo.cosine %[[ARG0]] : tensor<3xf32>
  // CHECK: %[[BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[COS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  %1 = mhlo.cosine %0 : tensor<4x3xf32>
  // CHECK: return %[[BCAST]]
  return %1 : tensor<4x3xf32>
}

// -----

// CHECK: @reorder_in_dim_2d_unary(%[[ARG0:.*]]: tensor<2x4xf32>) -> tensor<3x2x4xf32>
func.func @reorder_in_dim_2d_unary(%arg0: tensor<2x4xf32>) -> tensor<3x2x4xf32> {
  // CHECK: %[[LOG:.*]] = mhlo.log %[[ARG0]] : tensor<2x4xf32>
  // CHECK: %[[BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[LOG]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xf32>) -> tensor<3x2x4xf32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xf32>) -> tensor<3x2x4xf32>
  %1 = mhlo.log %0 : tensor<3x2x4xf32>
  // CHECK: return %[[BCAST]]
  return %1 : tensor<3x2x4xf32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_scalar_unary_diff_type(%[[ARG0:.*]]: tensor<complex<f32>>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>)
func.func @reorder_broadcast_in_dim_scalar_unary_diff_type(%arg0: tensor<complex<f32>>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) {
  // CHECK: %[[REAL:.*]] = mhlo.real %[[ARG0]] : (tensor<complex<f32>>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[REAL]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[IMAG:.*]] = mhlo.imag %[[ARG0]] : (tensor<complex<f32>>) -> tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[IMAG]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<complex<f32>>) -> tensor<1x8x8x64xcomplex<f32>>
  %1 = mhlo.real %0 : (tensor<1x8x8x64xcomplex<f32>>) -> tensor<1x8x8x64xf32>
  %2 = mhlo.imag %0 : (tensor<1x8x8x64xcomplex<f32>>) -> tensor<1x8x8x64xf32>
  return %1, %2: tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>
}

// -----

func.func @rng_normal(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<3x5xf32> {
  %shape = mhlo.constant dense<[3, 5]> : tensor<2xi64>
  %0 = "mhlo.rng"(%arg0, %arg1, %shape) {rng_distribution = #mhlo.rng_distribution<NORMAL>} : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}
// CHECK-LABEL: func.func @rng_normal
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG:     %{{.*}} = mhlo.constant dense<{{.*}}> : tensor<8xf32>
// CHECK-DAG:     %{{.*}} = mhlo.constant dense<{{.*}}> : tensor<8xf32>
// CHECK-DAG:     %{{.*}} = mhlo.constant dense<{{.*}}> : tensor<8xf32>
// CHECK:         %[[SIGMA:.+]] = "mhlo.broadcast"(%[[ARG1]]) {broadcast_sizes = dense<8> : tensor<1xi64>} : (tensor<f32>) -> tensor<8xf32>
//
//                mag = sigma * sqrt(-2.0 * log(u1)) where sqrt values are
//                constants.
//
// CHECK:         %[[MAG:.+]] = mhlo.multiply %[[SIGMA]], %{{.*}} : tensor<8xf32>
//
//                z0  = mag * cos(two_pi * u2) + mu;
//                z1  = mag * sin(two_pi * u2) + mu;
//
// CHECK:         %[[MU:.+]] = "mhlo.broadcast"(%[[ARG0]]) {broadcast_sizes = dense<8> : tensor<1xi64>} : (tensor<f32>) -> tensor<8xf32>
// CHECK:         %[[T1:.+]] = mhlo.multiply %[[MAG]], %{{.*}} : tensor<8xf32>
// CHECK:         %[[Z0:.+]] = mhlo.add %[[T1:.+]], %[[MU]] : tensor<8xf32>
// CHECK:         %[[T2:.+]] = mhlo.multiply %[[MAG]], %{{.*}} : tensor<8xf32>
// CHECK:         %[[Z1:.+]] = mhlo.add %[[T2:.+]], %[[MU]] : tensor<8xf32>
//
//                Concate and reshape the output.
// CHECK:         %[[CON:.+]] = "mhlo.concatenate"(%[[Z0]], %[[Z1]]) {dimension = 0 : i64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[CON]][0] [15] [1] : tensor<16xf32> to tensor<15xf32>
// CHECK:         %[[RES:.+]] = mhlo.reshape %[[SLICE]] : (tensor<15xf32>) -> tensor<3x5xf32>
// CHECK:         return %[[RES]]

// -----

func.func @mul_float_bool_cast(%arg0 : tensor<?xi1>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  %0 = mhlo.convert %arg0 : (tensor<?xi1>) -> tensor<?xf32>
  %1 = "mhlo.multiply"(%0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: @mul_float_bool_cast
// CHECK: %[[ZERO:.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[BTOF:.+]] = mhlo.convert %arg0 : (tensor<?xi1>) -> tensor<?xf32>
// CHECK: %[[FTOB:.+]] = mhlo.convert %[[BTOF]] : (tensor<?xf32>) -> tensor<?xi1>
// CHECK: %[[SHP:.+]] = shape.shape_of %[[BTOF]] : tensor<?xf32> -> tensor<1xindex>
// CHECK: %[[BROADCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[ZERO]], %[[SHP]]) {broadcast_dimensions = dense<> : tensor<0xi64>}
// CHECK: %[[SELECT:.+]] = mhlo.select %[[FTOB]], %arg1, %[[BROADCAST]]

// -----

func.func @mul_float_bool_cast_broadcast(%arg0: tensor<5xi1>, %arg1: tensor<5x6xf32>) -> tensor<5x6xf32> {
  %0 = mhlo.convert %arg0 : (tensor<5xi1>) -> tensor<5xf32>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<5xf32>) -> tensor<5x6xf32>
  %2 = mhlo.multiply %1, %arg1 : tensor<5x6xf32>
  return %2 : tensor<5x6xf32>
}

// CHECK-LABEL: @mul_float_bool_cast_broadcast
// CHECK: mhlo.select

// -----

func.func @mul_float_bool_cast_dyn_broadcast(%arg0: tensor<?xi1>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = mhlo.convert %arg0 : (tensor<?xi1>) -> tensor<?xf32>
    %1 = shape.shape_of %arg1 : tensor<?x?xf32> -> tensor<2xindex>
    %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
    %3 = mhlo.multiply %2, %arg1 : tensor<?x?xf32>
    return %3 : tensor<?x?xf32>
}

// CHECK-LABEL: @mul_float_bool_cast_dyn_broadcast
// CHECK: mhlo.select

// -----

// CHECK-LABEL: @dot_general_fuse_both_with_attrs
func.func @dot_general_fuse_both_with_attrs(%arg0: tensor<16x64x128xf16>, %arg1: tensor<16x128x3072xf16>) -> tensor<16x64x3072xf32> {
  %0 = mhlo.convert %arg0 : (tensor<16x64x128xf16>) -> tensor<16x64x128xf32>
  %1 = mhlo.convert %arg1 : (tensor<16x128x3072xf16>) -> tensor<16x128x3072xf32>
  // CHECK: "mhlo.dot_general"(%arg0, %arg1)
    // CHECK-SAME: dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    // CHECK-SAME: precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
    // CHECK-SAME: -> tensor<16x64x3072xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x128xf32>, tensor<16x128x3072xf32>) -> tensor<16x64x3072xf32>
  return %2 : tensor<16x64x3072xf32>
}

// -----

// CHECK-LABEL: @dot_general_fuse_one
func.func @dot_general_fuse_one(%arg0: tensor<16x64x128xf64>, %arg1: tensor<16x128x3072xf16>) -> tensor<16x64x3072xf32> {
  %0 = mhlo.convert %arg0 : (tensor<16x64x128xf64>) -> tensor<16x64x128xf32>
  %1 = mhlo.convert%arg1 : (tensor<16x128x3072xf16>) -> tensor<16x128x3072xf32>
  // CHECK: %[[CONVERT:.+]] = mhlo.convert %arg0
  // CHECK: "mhlo.dot_general"(%[[CONVERT]], %arg1)
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x128xf32>, tensor<16x128x3072xf32>) -> tensor<16x64x3072xf32>
  return %2 : tensor<16x64x3072xf32>
}

// -----

// CHECK-LABEL: @dot_basic
func.func @dot_basic(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf16>) -> tensor<4x4xf32> {
  %0 = mhlo.convert %arg0 : (tensor<4x4xf16>) -> tensor<4x4xf32>
  %1 = mhlo.convert %arg1 : (tensor<4x4xf16>) -> tensor<4x4xf32>
  // CHECK: %[[DOT:.+]] = "mhlo.dot"(%arg0, %arg1)
  %2 = "mhlo.dot"(%0, %1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: return %[[DOT]]
  return %2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @convolution
func.func @convolution(%arg0: tensor<16x32x256xbf16>, %arg1: tensor<1x256x256xbf16>) -> tensor<16x32x256xf32> {
  %cast = mhlo.convert %arg0 : (tensor<16x32x256xbf16>) -> tensor<16x32x256xf32>
  // CHECK: %[[CONV:.+]] = mhlo.convolution(%arg0, %arg1)
  // CHECK-SAME: -> tensor<16x32x256xf32>
  %0 = "mhlo.convolution"(%cast, %arg1) {
     batch_group_count = 1 : i64,
     dimension_numbers = #mhlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
     feature_group_count = 1 : i64,
     lhs_dilation = dense<1> : tensor<1xi64>,
     padding = dense<0> : tensor<1x2xi64>,
     precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
     rhs_dilation = dense<1> : tensor<1xi64>,
     window_strides = dense<1> : tensor<1xi64>
   } : (tensor<16x32x256xf32>, tensor<1x256x256xbf16>) -> tensor<16x32x256xf32>
  // CHECK: return %[[CONV]]
  func.return %0 : tensor<16x32x256xf32>
}

// -----

// CHECK-LABEL: @dynamic_dot_general
// This verifies non-crashing, the lowering to linalg happens elsewhere.
func.func @dynamic_dot_general(%arg1: tensor<?x1024x16x64xf32>, %arg2: tensor<?x1024x16x64xf32>) -> tensor<?x16x1024x1024xf32> {
  %2 = "mhlo.dot_general"(%arg2, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 2], rhs_batching_dimensions = [0, 2], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [3]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<?x1024x16x64xf32>, tensor<?x1024x16x64xf32>) -> tensor<?x16x1024x1024xf32>
  return %2 : tensor<?x16x1024x1024xf32>
}

