// RUN: iree-opt %s | FileCheck %s

// CHECK-LABEL: @foo.bar$baz-1
util.func @foo.bar$baz-1() {
  // CHECK: util.return
  util.return
}
