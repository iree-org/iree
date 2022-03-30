// RUN: iree-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @preserve_compiler_hints
func.func @preserve_compiler_hints() {
  // CHECK: %[[C:.+]] = arith.constant 2
  %c = arith.constant 2 : i32
  // CHECK: util.do_not_optimize(%[[C]])
  util.do_not_optimize(%c) : i32
  return
}
