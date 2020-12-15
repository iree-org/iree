// RUN: iree-tf-opt -iree-tf-convert-to-mhlo -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: func @f
func @f(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<3xf32>) {
  // CHECK: [[VAL0:%.+]] = mhlo.constant dense<2.000000e+00>
  // CHECK: [[VAL1:%.+]] = mhlo.constant dense<1.000000e+00>
  %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
  %1 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %4 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %5 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %6 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %7 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %8 = "tf.GreaterEqual"(%2, %4) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %9 = "tf.StridedSlice"(%5, %7, %5, %5) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %10 = "tf.SelectV2"(%0, %4, %9) {device = ""} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %11 = "tf.Range"(%4, %9, %6) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1xi32>
  %12 = "tf.Equal"(%10, %11) {device = "", incompatible_shape_error = true} : (tensor<i32>, tensor<1xi32>) -> tensor<1xi1>
  %13 = "tf.SelectV2"(%12, %2, %5) {device = ""} : (tensor<1xi1>, tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %14 = "tf.Sub"(%2, %6) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %15 = "tf.Maximum"(%14, %6) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %16 = "tf.Cast"(%15) {Truncate = false, device = ""} : (tensor<i32>) -> tensor<f32>
  %17 = "tf.SelectV2"(%8, %15, %1) {device = ""} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %18 = "tf.Cast"(%17) {Truncate = false, device = ""} : (tensor<i32>) -> tensor<i64>
  %19 = "tf.Range"(%3, %18, %3) {device = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1xi64>
  %20 = "tf.Cast"(%19) {Truncate = false, device = ""} : (tensor<1xi64>) -> tensor<1xf32>

  // CHECK: [[VAL2:%.+]] = "mhlo.reshape"(%arg0)
  %21 = "tf.ExpandDims"(%arg0, %4) {device = ""} : (tensor<f32>, tensor<i32>) -> tensor<1xf32>

  // CHECK: [[VAL3:%.+]] = "mhlo.reshape"(%arg1)
  %22 = "tf.ExpandDims"(%arg1, %4) {device = ""} : (tensor<f32>, tensor<i32>) -> tensor<1xf32>

  // CHECK: [[VAL4:%.+]] = mhlo.subtract [[VAL3]], [[VAL2]]
  %23 = "tf.Sub"(%22, %21) {device = ""} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: [[VAL5:%.+]] = mhlo.divide [[VAL4]], [[VAL0]]
  %24 = "tf.RealDiv"(%23, %16) {device = ""} : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>

  // CHECK: [[VAL6:%.+]] = mhlo.multiply [[VAL5]], [[VAL1]]
  %25 = "tf.Mul"(%24, %20) {device = ""} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: [[VAL7:%.+]] = mhlo.add [[VAL2]], [[VAL6]]
  %26 = "tf.AddV2"(%21, %25) {device = ""} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: [[VAL8:%.+]] = "mhlo.concatenate"([[VAL2]], [[VAL7]], [[VAL3]]) {dimension = 0 : i64}
  %27 = "tf.ConcatV2"(%21, %26, %22, %10) {device = ""} : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<i32>) -> tensor<3xf32>
  %28 = "tf.Slice"(%27, %7, %13) {device = ""} : (tensor<3xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xf32>
  %29 = "tf.Identity"(%28) {device = ""} : (tensor<3xf32>) -> tensor<3xf32>

  // CHECK: return [[VAL8]]
  return %29 : tensor<3xf32>
}
