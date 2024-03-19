// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-util-strip-debug-ops))' %s | FileCheck %s

// CHECK-LABEL: @stripAssert
util.func @stripAssert(%cond: i1) {
  // CHECK-NOT: cf.assert
  cf.assert %cond, "hello!"
  util.return
}
