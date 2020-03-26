// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @i32ToStringOp
func @i32ToStringOp(%arg0 : i32) -> !strings.string {
  // CHECK: "strings.i32_to_string"(%arg0) : (i32) -> !strings.string
  %0 = "strings.i32_to_string"(%arg0) : (i32) -> !strings.string
  return %0 : !strings.string
}

// -----

// CHECK-LABEL: @printOp
func @printOp(%arg0 : !strings.string) {
  // CHECK: "strings.print"(%arg0) : (!strings.string) -> ()
  "strings.print"(%arg0) : (!strings.string) -> ()
  return
}

// -----

// CHECK-LABEL: @toStringOp
func @toStringOp(%arg0 : !hal.buffer_view) -> !strings.string_tensor {
  // CHECK: "strings.to_string"(%arg0) : (!hal.buffer_view) -> !strings.string_tensor
  %0 = "strings.to_string"(%arg0) : (!hal.buffer_view) -> !strings.string_tensor
  return %0 : !strings.string_tensor
}

// -----

// CHECK-LABEL: @printTensorOp
func @printTensorOp(%arg0 : !strings.string_tensor) {
  // CHECK: "strings.print_tensor"(%arg0) : (!strings.string_tensor)
  "strings.print_tensor"(%arg0) : (!strings.string_tensor) -> ()
  return
}

