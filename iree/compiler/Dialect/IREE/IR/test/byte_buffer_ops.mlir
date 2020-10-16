// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @byte_buffer_constant
func @byte_buffer_constant() -> !iree.byte_buffer {
  // CHECK: = iree.byte_buffer.constant : !iree.byte_buffer = dense<[1, 2, 3]> : tensor<3xi32>
  %0 = iree.byte_buffer.constant : !iree.byte_buffer = dense<[1, 2, 3]> : tensor<3xi32>
  return %0 : !iree.byte_buffer
}
