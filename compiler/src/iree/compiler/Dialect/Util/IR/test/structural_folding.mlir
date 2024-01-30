// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// CHECK-NOT: util.initializer
util.initializer {
  util.return
}
