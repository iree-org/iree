// RUN: iree-tf-opt %s --iree-tf-verify-fully-converted --split-input-file -verify-diagnostics

// CHECK-LABEL: func.func @f
func.func @f() -> (tensor<i32>) {
  // CHECK: [[VAL0:%.+]] = mhlo.constant dense<3>
  %0 = mhlo.constant dense<3> : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// expected-error@below {{The following illegal operations still remain}}
func.func @f() -> (tensor<i32>) {
  // expected-error@+1 {{'tf.Const' op : illegal op still exists}}
  %0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// expected-error@below {{The following illegal operations still remain}}
func.func @f(%arg0 : tensor<i32>) -> (tensor<i32>) {
  // expected-error@+1 {{'tf.Const' op : illegal op still exists}}
  %0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  // expected-error@+1 {{'tf.Add' op : illegal op still exists}}
  %1 = "tf.Add"(%arg0, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}
