// RUN: iree-tf-opt --iree-tf-convert-to-tf-strings %s --split-input-file | IreeFileCheck %s

// CHECK-LABEL: func @as_string.tensor.f32
func @as_string.tensor.f32(%arg0: tensor<5xf32>) -> tensor<5x!tf.string> {
  // CHECK-DAG: [[VAL0:%.+]] = "tf_strings.to_string_tensor"(%arg0)
  %0 = "tf.AsString"(%arg0) {fill = ""} : (tensor<5xf32>) -> tensor<5x!tf.string>
  // CHECK: return [[VAL0]]
  return %0 : tensor<5x!tf.string>
}

// CHECK-LABEL: func @string_tensor_to_string
func @string_tensor_to_string(%arg0: tensor<5x!tf.string>) -> tensor<!tf.string> {
  // CHECK-DAG: [[VAL0:%.+]] = "tf_strings.string_tensor_to_string"(%arg0)
  %0 = "tf.StringFormat"(%arg0) : (tensor<5x!tf.string>) -> tensor<!tf.string>
  // CHECK: return [[VAL0]] : !tf_strings.string
  return %0 : tensor<!tf.string>
}

// CHECK-LABEL: func @printv2.tensor.string
func @printv2.tensor.string(%arg0: tensor<5x!tf.string>) {
  // CHECK-DAG: [[VAL0:%.+]] = "tf_strings.string_tensor_to_string"(%arg0)
  %0 = "tf.StringFormat"(%arg0) : (tensor<5x!tf.string>) -> tensor<!tf.string>
  // CHECK:     tf_strings.print"([[VAL0]])
  "tf.PrintV2"(%0) {_output_shapes = [], device = "", end = "\0A", output_stream = "stderr"} : (tensor<!tf.string>) -> ()
  return
}

// CHECK-LABEL: func @printv2.tensor.f32
func @printv2.tensor.f32(%arg0: tensor<5xf32>) {
  // CHECK-NEXT: [[VAL0:%.+]] = "tf_strings.to_string_tensor"(%arg0)
  // CHECK-DAG:  [[VAL1:%.+]] = "tf_strings.string_tensor_to_string"([[VAL0]])
  // CHECK:      "tf_strings.print"([[VAL1]])
  %0 = "tf.AsString"(%arg0) {fill = ""} : (tensor<5xf32>) -> tensor<5x!tf.string>
  %1 = "tf.StringFormat"(%0) : (tensor<5x!tf.string>) -> tensor<!tf.string>
  "tf.PrintV2"(%1) {_output_shapes = [], device = "", end = "\0A", output_stream = "stderr"} : (tensor<!tf.string>) -> ()
  return
}

// CHECK-LABEL: func @gatherv2
func @gatherv2(%arg0: tensor<5x!tf.string>, %arg1: tensor<3xi32>) {
  %0 = "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>

  // CHECK-DAG: [[VAL0:%.+]] = "tf_strings.gather"(%arg0, %arg1)
  %1 = "tf.GatherV2"(%arg0, %arg1, %0) {batch_dims = 0 : i64, device = ""} : (tensor<5x!tf.string>, tensor<3xi32>, tensor<i32>) -> tensor<3x!tf.string>
  return
}
