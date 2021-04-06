// RUN: iree-tf-opt -split-input-file -verify-diagnostics -pass-pipeline='func(iree-tf-strip-asserts)' %s | IreeFileCheck %s

// CHECK-LABEL: @asserts
// CHECK-NOT: tf.Assert
func @asserts(%arg0 : tensor<*xi1>, %arg1 : tensor<!tf.string>,
      %arg2 : tensor<!tf.string>, %arg3 : tensor<!tf.string>,
      %arg4 : tensor<i32>, %arg5 : tensor<!tf.string>, %arg6 : tensor<i32>) {
  "tf.Assert"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6)
    : (tensor<*xi1>, tensor<!tf.string>, tensor<!tf.string>, tensor<!tf.string>,
       tensor<i32>, tensor<!tf.string>, tensor<i32>) -> ()
  return
}
