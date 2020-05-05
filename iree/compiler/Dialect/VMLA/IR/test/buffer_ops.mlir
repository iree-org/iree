// Tests the printing/parsing of the VMLA dialect buffer ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: vmla_buffer_const
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9$._-]+]]
func @vmla_buffer_const(%value : !iree.byte_buffer) {
  // CHECK: vmla.buffer.const %[[VALUE]] : !iree.byte_buffer -> !vmla.buffer
  %result = vmla.buffer.const %value : !iree.byte_buffer -> !vmla.buffer
  return
}

// -----

// CHECK-LABEL: vmla_buffer_alloc
// CHECK-SAME: %[[LENGTH:[a-zA-Z0-9$._-]+]]
func @vmla_buffer_alloc(%byte_length : index) {
  // CHECK: vmla.buffer.alloc byte_length = %[[LENGTH]] : !vmla.buffer
  %result = vmla.buffer.alloc byte_length = %byte_length : !vmla.buffer
  return
}

// -----

// CHECK-LABEL: vmla_buffer_clone
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
func @vmla_buffer_clone(%src : !vmla.buffer) {
  // CHECK: vmla.buffer.clone %[[SRC]] : !vmla.buffer
  %result = vmla.buffer.clone %src : !vmla.buffer
  return
}

// -----

// CHECK-LABEL: vmla_buffer_byte_length
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9$._-]+]]
func @vmla_buffer_byte_length(%value : !vmla.buffer) {
  // CHECK: vmla.buffer.byte_length %[[VALUE]] : index
  %result = vmla.buffer.byte_length %value : index
  return
}

// -----

// CHECK-LABEL: vmla_buffer_view
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[OFFSET:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[LENGTH:[a-zA-Z0-9$._-]+]]
func @vmla_buffer_view(%src : !vmla.buffer,
                       %byte_offset : index,
                       %byte_length : index) {
  // CHECK:      vmla.buffer.view %[[SRC]][%[[OFFSET]]],
  // CHECK-SAME: byte_length = %[[LENGTH]] : !vmla.buffer
  %result = vmla.buffer.view %src[%byte_offset],
                             byte_length = %byte_length : !vmla.buffer
  return
}

// -----

// CHECK-LABEL: vmla_buffer_copy
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SRC_OFFSET:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_OFFSET:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[LENGTH:[a-zA-Z0-9$._-]+]]
func @vmla_buffer_copy(%src : !vmla.buffer,
                       %src_byte_offset : index,
                       %dst : !vmla.buffer,
                       %dst_byte_offset : index,
                       %byte_length : index) {
  // CHECK:      vmla.buffer.copy %[[SRC]][%[[SRC_OFFSET]]],
  // CHECK-SAME: out %[[DST]][%[[DST_OFFSET]]], byte_length = %[[LENGTH]]
  vmla.buffer.copy %src[%src_byte_offset],
                   out %dst[%dst_byte_offset], byte_length = %byte_length
  return
}

// -----

// CHECK-LABEL: vmla_buffer_fill
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
func @vmla_buffer_fill(%src : !vmla.buffer, %dst : !vmla.buffer) {
  // CHECK: vmla.buffer.fill %[[VALUE]], out %[[DST]]
  vmla.buffer.fill %src, out %dst
  return
}

// -----

// CHECK-LABEL: vmla_buffer_load_i32
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[OFFSET:[a-zA-Z0-9$._-]+]]
func @vmla_buffer_load_i32(%src : !vmla.buffer, %byte_offset : index) {
  // CHECK: vmla.buffer.load.i32 %[[SRC]][%[[OFFSET]]] : i32
  %result = vmla.buffer.load.i32 %src[%byte_offset] : i32
  return
}
