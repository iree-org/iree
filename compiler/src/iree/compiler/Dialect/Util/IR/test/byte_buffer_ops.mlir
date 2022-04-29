// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | FileCheck %s

// CHECK-LABEL: @byte_buffer_constant
func.func @byte_buffer_constant() -> !util.byte_buffer {
  // CHECK: = util.byte_buffer.constant : !util.byte_buffer = dense<[1, 2, 3]> : tensor<3xi32>
  %0 = util.byte_buffer.constant : !util.byte_buffer = dense<[1, 2, 3]> : tensor<3xi32>
  return %0 : !util.byte_buffer
}
