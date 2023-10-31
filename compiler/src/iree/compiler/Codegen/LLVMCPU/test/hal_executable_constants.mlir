// RUN: iree-opt --iree-convert-to-llvm --split-input-file %s | FileCheck %s

// CHECK-DAG: llvm.mlir.global internal @__constant_ordinal_foo() {{.+}}hal.executable.constant.key = "foo"{{.+}} : i32
// CHECK-LABEL: llvm.func @constant_values
func.func @constant_values() {
  // CHECK: %[[FOO_ORDINAL_PTR:.+]] = llvm.mlir.addressof @__constant_ordinal_foo : !llvm.ptr
  // CHECK: %[[FOO_ORDINAL:.+]] = llvm.load %[[FOO_ORDINAL_PTR]]
  // CHECK: %[[FOO_PTR:.+]] = llvm.getelementptr %{{.+}}[%[[FOO_ORDINAL]]]
  // CHECK: %[[FOO:.+]] = llvm.load %[[FOO_PTR]]
  %v0 = hal.executable.constant.load "foo" : i32
  // CHECK: llvm.call @sink(%[[FOO]])
  llvm.call @sink(%v0) : (i32) -> ()
  return
}
llvm.func @sink(%arg0: i32) {
  llvm.return
}
