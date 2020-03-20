// RUN: iree-tf-opt --convert-tensorflow-to-tf-strings %s --split-input-file | IreeFileCheck %s

// CHECK-LABEL: func @as_string.i32
func @as_string.i32(%arg0: tensor<i32>) -> tensor<!tf.string> {
  // CHECK-NEXT: [[VAL0:%.+]] = "tf_strings.to_string"(%arg0)
  %0 = "tf.AsString"(%arg0) {fill = ""} : (tensor<i32>) -> tensor<!tf.string>

  // CHECK-NEXT: return [[VAL0]]
  return %0 : tensor<!tf.string>
}

// CHECK-LABEL: func @print.i32
func @print.i32(%arg0: tensor<i32>) {
  // CHECK-NEXT: [[VAL0:%.+]] = "tf_strings.to_string"(%arg0)
  %0 = "tf.AsString"(%arg0) {fill = ""} : (tensor<i32>) -> tensor<!tf.string>

  // CHECK-NEXT: "tf_strings.print"([[VAL0]])
  "tf.PrintV2"(%0) {_output_shapes = [], device = "", end = "\0A", output_stream = "stderr"} : (tensor<!tf.string>) -> ()
  return
}

// CHECK-LABEL: func @print.string
func @print.string(%arg0: tensor<!tf.string>) {
  // CHECK-NEXT: "tf_strings.print"(%arg0)
  "tf.PrintV2"(%arg0) {_output_shapes = [], device = "", end = "\0A", output_stream = "stderr"} : (tensor<!tf.string>) -> ()
  return
}

// CHECK-LABEL: func @call.test
func @call.test(%arg0: tensor<i32>) {
  // CHECK-NEXT: [[VAL0:%.+]] = call @as_string.i32(%arg0)
  %0 = call @as_string.i32(%arg0) : (tensor<i32>) -> (tensor<!tf.string>)

  // CHECK-NEXT: call @print.string([[VAL0]])
  call @print.string(%0) : (tensor<!tf.string>) -> ()
  return
}
