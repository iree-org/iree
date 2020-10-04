// RUN: iree-tf-opt --iree-convert-to-hal %s --split-input-file | IreeFileCheck %s

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
  // CHECK-DAG: [[VAL0:%.+]]  = hal.buffer_view.create %arg0{{.*}}
  // CHECK-DAG: [[VAL1:%.+]] = "strings.to_string_tensor"([[VAL0]])
  %0 = "tf_strings.to_string_tensor"(%arg0) : (tensor<5xf32>) -> tensor<5x!tf_strings.string>

  // CHECK: return [[VAL1]]
  return %0 : tensor<5x!tf_strings.string>
}

// CHECK-LABEL: @string_tensor_to_string
func @string_tensor_to_string(%arg0 : tensor<!tf_strings.string>) -> !tf_strings.string {
  // CHECK-DAG: [[VAL0:%.+]] = "strings.string_tensor_to_string"(%arg0)
  %0 = "tf_strings.string_tensor_to_string"(%arg0) : (tensor<!tf_strings.string>) -> (!tf_strings.string)

  // CHECK: return [[VAL0]]
  return %0 : !tf_strings.string
}
