
// RUN: iree-opt -split-input-file -pass-pipeline='builtin.func(iree-flow-strip-signedness)' %s | IreeFileCheck %s

// CHECK-LABEL: @strip_signedness_arg
// CHECK-SAME: tensor<4xi8>
func @strip_signedness_arg(%arg0 : tensor<4xui8>) -> (tensor<4xui8>) {
    // CHECK: return
    // CHECK-SAME: tensor<4xi8>
    return %arg0 : tensor<4xui8>
}

// ----

// CHECK-LABEL: @strip_signedness_const
func @strip_signedness_const() -> (tensor<4xi8>) {
    // CHECK: constant
    // CHECK-SAME: tensor<4xi8>
    %0 = constant dense<[0, 2, 3, 7]> : tensor<4xi8>
    // CHECK: return
    // CHECK-SAME: tensor<4xi8>
    return %0 : tensor<4xi8>
}
