// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-NOT: util.initializer
util.initializer {
  util.initializer.return
}
