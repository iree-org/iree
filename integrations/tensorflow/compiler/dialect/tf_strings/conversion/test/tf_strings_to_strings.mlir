// RUN: iree-tf-opt --convert-tf_strings-to-strings %s --split-input-file | IreeFileCheck %s

// CHECK-LABEL: @to_string.i32
func @to_string.i32(%arg0 : i32) -> (!tf_strings.string) {
  // CHECK-DAG: [[VAL0:%.+]] = "strings.i32_to_string"(%arg0)
  %0 = "tf_strings.to_string"(%arg0) : (i32) -> (!tf_strings.string)

  // CHECK: return [[VAL0]] : !strings.string
  return %0 : !tf_strings.string
}

// CHECK-LABEL: @print.i32
func @print.i32(%arg0 : i32) {
  // CHECK-DAG: [[VAL0:%.+]] = "strings.i32_to_string"(%arg0)
  %0 = "tf_strings.to_string"(%arg0) : (i32) -> (!tf_strings.string)

  // CHECK: "strings.print"([[VAL0]]) : (!strings.string) -> ()
  "tf_strings.print"(%0) : (!tf_strings.string) -> ()
  return
}
