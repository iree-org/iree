// RUN: iree-opt -split-input-file %s | iree-opt -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: @buffer_view_create
func @buffer_view_create(%arg0: !hal.buffer, %arg1: index, %arg2: index) -> !hal.buffer_view {
  %c1 = arith.constant 1 : i32
  %c32 = arith.constant 32 : i32
  // CHECK: %view = hal.buffer_view.create
  // CHECK-SAME: buffer(%arg0 : !hal.buffer)
  // CHECK-SAME: shape([%arg1, %arg2])
  // CHECK-SAME: type(%c32_i32)
  // CHECK-SAME: encoding(%c1_i32) : !hal.buffer_view
  %view = hal.buffer_view.create buffer(%arg0 : !hal.buffer)
                                 shape([%arg1, %arg2])
                                 type(%c32)
                                 encoding(%c1) : !hal.buffer_view
  return %view : !hal.buffer_view
}

// -----

// CHECK-LABEL: @buffer_view_buffer
func @buffer_view_buffer(%arg0: !hal.buffer_view) -> !hal.buffer {
  // CHECK: %buffer = hal.buffer_view.buffer<%arg0 : !hal.buffer_view> : !hal.buffer
  %buffer = hal.buffer_view.buffer<%arg0 : !hal.buffer_view> : !hal.buffer
  return %buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @buffer_view_byte_length
func @buffer_view_byte_length(%arg0: !hal.buffer_view) -> index {
  // CHECK: %len = hal.buffer_view.byte_length<%arg0 : !hal.buffer_view> : index
  %len = hal.buffer_view.byte_length<%arg0 : !hal.buffer_view> : index
  return %len : index
}

// -----

// CHECK-LABEL: @buffer_view_shape_queries
func @buffer_view_shape_queries(%arg0: !hal.buffer_view) -> (index, index, index, index) {
  // CHECK: %{{.+}} = hal.buffer_view.rank<%arg0 : !hal.buffer_view> : index
  %0 = hal.buffer_view.rank<%arg0 : !hal.buffer_view> : index
  // CHECK: %{{.+}} = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  // CHECK: %{{.+}}:2 = hal.buffer_view.dims<%arg0 : !hal.buffer_view> : index, index
  %2, %3 = hal.buffer_view.dims<%arg0 : !hal.buffer_view> : index, index
  return %0, %1, %2, %3 : index, index, index, index
}
