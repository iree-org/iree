// RUN: strings-opt -split-input-file %s | strings-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @i32ToStringOp
func @i32ToStringOp(%arg0 : i32) -> !iree.ref<!strings.string> {
  // CHECK: "strings.i32_to_string"(%arg0) : (i32) -> !iree.ref<!strings.string>
  %0 = "strings.i32_to_string"(%arg0) : (i32) -> !iree.ref<!strings.string>
  return %0 : !iree.ref<!strings.string>
}

// -----

// CHECK-LABEL: @printOp
func @printOp(%arg0 : !iree.ref<!strings.string>) {
  // CHECK: "strings.print"(%arg0) : (!iree.ref<!strings.string>) -> ()
  "strings.print"(%arg0) : (!iree.ref<!strings.string>) -> ()
  return
}

