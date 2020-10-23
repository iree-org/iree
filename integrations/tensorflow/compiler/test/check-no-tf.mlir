// RUN: iree-tf-opt %s -iree-check-no-tf -split-input-file -verify-diagnostics

// CHECK-LABEL: func @f
func @f() -> (tensor<i32>) {
  // CHECK: [[VAL0:%.+]] = mhlo.constant dense<3>
  %0 = mhlo.constant dense<3> : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// expected-error@+3 {{'tf.Const' op still exists}}
// expected-error@below {{The following Tensorflow operations still remain}}
func @f() -> (tensor<i32>) {
  %0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// expected-error@+4 {{'tf.Const' op still exists}}
// expected-error@+4 {{'tf.Add' op still exists}}
// expected-error@below {{The following Tensorflow operations still remain}}
func @f(%arg0 : tensor<i32>) -> (tensor<i32>) {
  %0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Add"(%arg0, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}
