// RUN: strings-opt %s -iree-vm-conversion -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @i32ToStringOp
func @i32ToStringOp(%arg0 : i32) -> !strings.string {
  // CHECK: vm.call @strings.i32_to_string(%arg0) : (i32) -> !iree.ref<!strings.string>
  %0 = "strings.i32_to_string"(%arg0) : (i32) -> !strings.string
  return %0 : !strings.string
}

// CHECK: vm.import @strings.i32_to_string

// -----

// CHECK-LABEL: @printOp
func @printOp(%arg0 : !strings.string) {
  // CHECK: vm.call @strings.print(%arg0) : (!iree.ref<!strings.string>)
  "strings.print"(%arg0) : (!strings.string) -> ()
  return
}

// CHECK: vm.import @strings.print
