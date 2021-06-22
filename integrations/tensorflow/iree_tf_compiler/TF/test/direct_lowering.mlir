// RUN: iree-tf-opt -iree-tf-convert-to-mhlo -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: @expand_dims
func @expand_dims(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?x1xf32> {
  %axis = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> (tensor<i32>)
  %0 = "tf.ExpandDims"(%arg0, %axis) : (tensor<?x?x?xf32>, tensor<i32>) -> (tensor<?x?x?x1xf32>)
  return %0 : tensor<?x?x?x1xf32>
}

// -----
// CHECK-LABEL: @expand_dims
func @expand_dims(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %axis = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> (tensor<i32>)
  %0 = "tf.ExpandDims"(%arg0, %axis) : (tensor<?x?x?xf32>, tensor<i32>) -> (tensor<?x?x?x?xf32>)
  return %0 : tensor<?x?x?x?xf32>
}
