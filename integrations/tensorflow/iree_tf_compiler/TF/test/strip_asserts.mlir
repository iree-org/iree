// RUN: iree-tf-opt --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(func.func(iree-tf-strip-asserts))' %s | FileCheck %s

// CHECK-LABEL: @asserts
// CHECK-NOT: tf.Assert
func.func @asserts(%arg0 : tensor<*xi1>, %arg1 : tensor<!tf_type.string>,
      %arg2 : tensor<!tf_type.string>, %arg3 : tensor<!tf_type.string>,
      %arg4 : tensor<i32>, %arg5 : tensor<!tf_type.string>, %arg6 : tensor<i32>) {
  "tf.Assert"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6)
    : (tensor<*xi1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>,
       tensor<i32>, tensor<!tf_type.string>, tensor<i32>) -> ()
  return
}
