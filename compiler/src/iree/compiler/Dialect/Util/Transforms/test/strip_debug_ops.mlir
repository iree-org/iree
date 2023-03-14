// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-util-strip-debug-ops))' %s | FileCheck %s

// CHECK-LABEL: @stripAssert
func.func @stripAssert(%cond: i1) {
  // CHECK-NOT: cf.assert
  cf.assert %cond, "hello!"
  return
}
