// RUN: iree-tf-opt %s -iree-check-no-tf -split-input-file -verify-diagnostics

// CHECK-LABEL: func @f
func @f() -> (tensor<i32>) {
  // CHECK: [[VAL0:%.+]] = mhlo.constant dense<3>
  %0 = mhlo.constant dense<3> : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// expected-error@below {{The following operations cannot be legalized: tf.Const (count: 1). These legalization failure(s) may be due to missing TF to HLO lowerings and/or unsupported attributes, etc.}}
func @f() -> (tensor<i32>) {
  // expected-error@+1 {{'tf.Const' op is not legalizable}}
  %0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  return %0 : tensor<i32>
}
