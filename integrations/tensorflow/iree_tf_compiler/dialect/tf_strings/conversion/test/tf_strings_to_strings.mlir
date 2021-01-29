// RUN: iree-tf-opt --iree-tf-strings-convert-to-strings  %s --split-input-file | IreeFileCheck %s

// CHECK-LABEL: @i32_to_string
func @i32_to_string(%arg0 : i32) -> !tf_strings.string {
  // CHECK-DAG: [[VAL0:%.+]] = "strings.i32_to_string"(%arg0)
  %0 = "tf_strings.to_string"(%arg0) : (i32) -> (!tf_strings.string)

  // CHECK: return [[VAL0]]
  return %0 : !tf_strings.string
}

// CHECK-LABEL: @print_string
func @print_string(%arg0 : !tf_strings.string) {
  // CHECK-DAG: "strings.print"(%arg0)
  "tf_strings.print"(%arg0) : (!tf_strings.string) -> ()

  // CHECK: return
  return
}

// CHECK-LABEL: @to_string_tensor.f32
func @to_string_tensor.f32(%arg0 : tensor<5xf32>) -> tensor<5x!tf_strings.string> {
  // CHECK-DAG: [[VAL0:%.+]] = "strings.to_string_tensor"(%arg0)
  %0 = "tf_strings.to_string_tensor"(%arg0) : (tensor<5xf32>) -> tensor<5x!tf_strings.string>

  // CHECK: return [[VAL0]]
  return %0 : tensor<5x!tf_strings.string>
}

// CHECK-LABEL: @string_tensor_to_string
func @string_tensor_to_string(%arg0 : tensor<!tf_strings.string>) -> !tf_strings.string {
  // CHECK-DAG: [[VAL0:%.+]] = "strings.string_tensor_to_string"(%arg0)
  %0 = "tf_strings.string_tensor_to_string"(%arg0) : (tensor<!tf_strings.string>) -> (!tf_strings.string)

  // CHECK: return [[VAL0]]
  return %0 : !tf_strings.string
}

// CHECK-LABEL: @gather
func @gather(%arg0: tensor<5x!tf_strings.string>, %arg1: tensor<3xi32>) -> tensor<3x!tf_strings.string> {
  // CHECK-DAG: [[VAL0:%.+]] = "strings.gather"(%arg0, %arg1)
  %0 = "tf_strings.gather"(%arg0, %arg1) : (tensor<5x!tf_strings.string>, tensor<3xi32>) -> tensor<3x!tf_strings.string>

  // CHECK: return [[VAL0]]
  return %0 : tensor<3x!tf_strings.string>
}
