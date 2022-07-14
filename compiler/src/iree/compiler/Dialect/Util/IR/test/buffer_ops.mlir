// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @buffer_constant
func.func @buffer_constant() -> !util.buffer {
  // CHECK: = util.buffer.constant : !util.buffer = dense<[1, 2, 3]> : tensor<3xi32>
  %0 = util.buffer.constant : !util.buffer = dense<[1, 2, 3]> : tensor<3xi32>
  return %0 : !util.buffer
}
