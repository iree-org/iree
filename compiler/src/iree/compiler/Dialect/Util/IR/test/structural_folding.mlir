// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-NOT: util.initializer
util.initializer {
  util.return
}
