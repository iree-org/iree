// RUN: strings-opt -split-input-file %s | strings-opt -split-input-file | IreeFileCheck %s

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

