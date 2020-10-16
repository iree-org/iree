// RUN: iree-opt -iree-convert-to-hal %s --split-input-file | IreeFileCheck %s

// CHECK-LABEL: @preserve_compiler_hints
func @preserve_compiler_hints() {
  // CHECK: iree.do_not_optimize()
  iree.do_not_optimize()

  // CHECK: %[[C:.+]] = constant 2
  %c = constant 2 : i32
  // CHECK: iree.do_not_optimize(%[[C]])
  iree.do_not_optimize(%c) : i32
  return
}
