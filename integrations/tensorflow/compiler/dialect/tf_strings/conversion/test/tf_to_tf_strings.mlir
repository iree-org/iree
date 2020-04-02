// RUN: iree-tf-opt --convert-tensorflow-to-tf-strings %s --split-input-file | IreeFileCheck %s

// CHECK-LABEL: func @as_string.tensor.f32
func @as_string.tensor.f32(%arg0: tensor<5xf32>) -> tensor<5x!tf.string> {
  // CHECK-DAG: [[VAL0:%.+]] = "tf_strings.to_string_tensor"(%arg0)
  %0 = "tf.AsString"(%arg0) {fill = ""} : (tensor<5xf32>) -> tensor<5x!tf.string>
  // CHECK: return [[VAL0]]
  return %0 : tensor<5x!tf.string>
}

// CHECK-LABEL: func @printv2.tensor.string
func @printv2.tensor.string(%arg0: tensor<5x!tf.string>) {
  // CHECK-DAG: [[VAL0:%.+]] = "tf_strings.string_tensor_to_string"(%arg0)
  // CHECK:     tf_strings.print"([[VAL0]])
  "tf.PrintV2"(%arg0) {_output_shapes = [], device = "", end = "\0A", output_stream = "stderr"} : (tensor<5x!tf.string>) -> ()
  return
}

// CHECK-LABEL: func @printv2.tensor.f32
func @printv2.tensor.f32(%arg0: tensor<5xf32>) {
  // CHECK-NEXT: [[VAL0:%.+]] = "tf_strings.to_string_tensor"(%arg0)
  // CHECK-DAG:  [[VAL1:%.+]] = "tf_strings.string_tensor_to_string"([[VAL0]])
  // CHECK:      "tf_strings.print"([[VAL1]])
  %0 = "tf.AsString"(%arg0) {fill = ""} : (tensor<5xf32>) -> tensor<5x!tf.string>
  "tf.PrintV2"(%0) {_output_shapes = [], device = "", end = "\0A", output_stream = "stderr"} : (tensor<5x!tf.string>) -> ()
  return
}

