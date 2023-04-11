// RUN: iree-compile --iree-hal-target-backends=vmvx %s | iree-run-module --module=- --function=null_values --input="(null)" | FileCheck %s

// CHECK-LABEL: EXEC @null_values
func.func @null_values(%arg0: !vm.list<i32>) -> (i32, !vm.list<i32>) {
  %c123 = arith.constant 123 : i32
  return %c123, %arg0 : i32, !vm.list<i32>
}
// CHECK: result[0]: i32=123
// CHECK: result[1]: (null)
