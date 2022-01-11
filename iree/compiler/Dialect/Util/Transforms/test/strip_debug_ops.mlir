// RUN: iree-opt -split-input-file -pass-pipeline='builtin.func(iree-util-strip-debug-ops)' %s | IreeFileCheck %s

// CHECK-LABEL: @stripAssert
func @stripAssert(%cond: i1) {
  // CHECK-NOT: assert
  assert %cond, "hello!"
  return
}
