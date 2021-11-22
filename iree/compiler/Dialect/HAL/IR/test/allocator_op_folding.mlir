// RUN: iree-opt -split-input-file -canonicalize -cse %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @allocator_constant_buffer
// CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator
func @allocator_constant_buffer(%allocator: !hal.allocator) -> !hal.buffer {
  //      CHECK: %[[RODATA:.+]] = util.byte_buffer.constant : !util.byte_buffer = dense<123> : tensor<4x4xi32>
  // CHECK-NEXT: %[[BUFFER:.+]] = hal.allocator.map<%[[ALLOCATOR]] : !hal.allocator>
  // CHECK-SAME:   source(%[[RODATA]] : !util.byte_buffer)[%c0, %c-1]
  // CHECK-SAME:   type("HostVisible|DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage("Constant|Transfer|Mapping|Dispatch")
  // CHECK-SAME:   : !hal.buffer
  %ref = hal.allocator.constant<%allocator : !hal.allocator>
         type(DeviceLocal) usage(Transfer) : !hal.buffer =
         dense<123> : tensor<4x4xi32>
  // CHECK-NEXT: return %[[BUFFER]]
  return %ref : !hal.buffer
}

// -----

// CHECK-LABEL: @allocator_constant_buffer_view
// CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator
func @allocator_constant_buffer_view(%allocator: !hal.allocator) -> !hal.buffer_view {
  //      CHECK: %[[RODATA:.+]] = util.byte_buffer.constant : !util.byte_buffer = dense<123> : tensor<4x4xi32>
  // CHECK-NEXT: %[[BUFFER:.+]] = hal.allocator.map<%[[ALLOCATOR]] : !hal.allocator>
  // CHECK-SAME:   source(%[[RODATA]] : !util.byte_buffer)[%c0, %c-1]
  // CHECK-SAME:   type("HostVisible|DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage("Constant|Transfer|Mapping|Dispatch")
  // CHECK-SAME:   : !hal.buffer
  // CHECK-NEXT: %[[VIEW:.+]] = hal.buffer_view.create
  // CHECK-SAME:     buffer(%[[BUFFER]] : !hal.buffer)
  // CHECK-SAME:     shape([%c4, %c4])
  // CHECK-SAME:     type(%c268435488_i32)
  // CHECK-SAME:     encoding(%c1_i32) : !hal.buffer_view
  %ref = hal.allocator.constant<%allocator : !hal.allocator>
         type(DeviceLocal) usage(Transfer) : !hal.buffer_view =
         dense<123> : tensor<4x4xi32>
  // CHECK-NEXT: return %[[VIEW]]
  return %ref : !hal.buffer_view
}

// -----

// CHECK-LABEL: @allocator_pack_zero_offset
func @allocator_pack_zero_offset(%allocator: !hal.allocator, %size : index) -> index {
  // CHECK-NOT: constant 0
  %base_offset = arith.constant 0 : index
  // CHECK: hal.allocator.pack<{{.+}}> slices({
  %total_length, %offset_0, %offset_1 =
      hal.allocator.pack<%allocator : !hal.allocator>
        offset(%base_offset)
        slices({
          [0, 4] = %size,
          [1, 2] = %size,
        }) : index
  return %total_length : index
}

// -----

// A pack with no slices folds to a zero-length slab.

// CHECK-LABEL: @allocator_pack_no_slices
func @allocator_pack_no_slices(%allocator: !hal.allocator) -> index {
  // CHECK-NEXT: %[[ZERO_LENGTH:.+]] = arith.constant 0
  %total_length =
      hal.allocator.pack<%allocator : !hal.allocator> slices({}) : index
  // CHECK-NEXT: return %[[ZERO_LENGTH]]
  return %total_length : index
}

// -----

// A pack with a single slices folds to just that slice.

// CHECK-LABEL: @allocator_pack_one_slice
// CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator,
// CHECK-SAME: %[[OFFSET:.+]]: index,
// CHECK-SAME: %[[SIZE:.+]]: index
func @allocator_pack_one_slice(%allocator: !hal.allocator, %offset: index, %size: index) -> (index, index) {
  // CHECK-NOT: hal.allocator.pack
  %total_length, %offset_0 =
      hal.allocator.pack<%allocator : !hal.allocator>
        offset(%offset)
        slices({
          [0, 4] = %size
        }) : index
  // CHECK: return %[[SIZE]], %[[OFFSET]]
  return %total_length, %offset_0 : index, index
}

// -----

// A constant zero offset operand gets dropped.

// CHECK-LABEL: @allocator_pack_drop_zero_offset
func @allocator_pack_drop_zero_offset(%allocator: !hal.allocator, %size : index) -> (index, index, index) {
  // CHECK-NEXT: = hal.allocator.pack<{{.+}}> slices({
  %base_offset = arith.constant 0 : index
  %total_length, %offset_0, %offset_1 =
      hal.allocator.pack<%allocator : !hal.allocator>
        offset(%base_offset)
        slices({
          [0, 4] = %size,
          [1, 2] = %size,
        }) : index
  return %total_length, %offset_0, %offset_1 : index, index, index
}

// -----

// A base offset operand gets propagated to returned values.

// CHECK-LABEL: @allocator_pack_propagate_base_offset
// CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator,
// CHECK-SAME: %[[BASE_OFFSET:.+]]: index,
// CHECK-SAME: %[[SIZE:.+]]: index
func @allocator_pack_propagate_base_offset(%allocator: !hal.allocator, %base_offset: index, %size : index) -> (index, index, index) {
  // CHECK-NEXT: %[[PACKED:.+]]:3 =
  %total_length, %offset_0, %offset_1 =
      // CHECK-SAME: hal.allocator.pack<{{.+}}> slices({
      hal.allocator.pack<%allocator : !hal.allocator>
        offset(%base_offset)
        slices({
          [0, 4] = %size,
          [1, 2] = %size,
        }) : index
  //      CHECK: %[[ADJUSTED_0:.+]] = arith.addi %[[BASE_OFFSET]], %[[PACKED]]#1
  // CHECK-NEXT: %[[ADJUSTED_1:.+]] = arith.addi %[[BASE_OFFSET]], %[[PACKED]]#2
  // CHECK-NEXT: return %[[PACKED]]#0, %[[ADJUSTED_0]], %[[ADJUSTED_1]]
  return %total_length, %offset_0, %offset_1 : index, index, index
}

// -----

// Intervals should be sorted.

// CHECK-LABEL: @allocator_pack_sort_intervals
// CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator,
// CHECK-SAME: %[[SIZE:.+]]: index
func @allocator_pack_sort_intervals(%allocator: !hal.allocator, %size : index) -> (index, index, index) {
  // CHECK-NEXT: %[[PACKED:.+]]:3 =
  %total_length, %offset_0, %offset_1 =
      // CHECK-SAME: hal.allocator.pack<{{.+}}> slices({
      hal.allocator.pack<%allocator : !hal.allocator>
        slices({
          // CHECK-NEXT: [0, 4] = %[[SIZE]],
          // CHECK-NEXT: [1, 2] = %[[SIZE]]
          [1, 2] = %size,
          [0, 4] = %size,
        }) : index
  // CHECK: return %[[PACKED]]#0, %[[PACKED]]#2, %[[PACKED]]#1
  return %total_length, %offset_0, %offset_1 : index, index, index
}
