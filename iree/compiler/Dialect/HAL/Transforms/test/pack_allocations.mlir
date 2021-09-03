// RUN: iree-opt -split-input-file -iree-hal-pack-allocations -cse -canonicalize %s | IreeFileCheck %s

module attributes {
  hal.device.targets = [
    #hal.device.target<"cpu", {
      buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 16, max_buffer_range = 1073741824, min_buffer_range_alignment = 16>
    }>
  ]
} {

// CHECK-LABEL: @packStatic
// CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator
func @packStatic(%allocator: !hal.allocator) ->
    (index, index, index, index, index, index, index) {
  %c100 = constant 100 : index
  %c200 = constant 200 : index
  %t:7 = hal.allocator.pack<%allocator : !hal.allocator> slices({
    [0, 1] = %c100,  // +0
    [1, 2] = %c100,  // +112 (100 align 16)
    [2, 3] = %c100,  // +0 (reuse [0, 1])
    [0, 4] = %c200,  // +224 (after 112 + 112; end align 16)
    [5, 6] = %c200,  // +0 (reuse [0, 1]/[2, 3])
    [5, 8] = %c100,  // +208 (after 200 align 16)
  }) : index
  // 224 + 200 align 16 = 432 total bytes required
  // CHECK: return %c432
  // CHECK-SAME: %c0, %c112, %c0, %c224, %c0, %c208
  return %t#0, %t#1, %t#2, %t#3, %t#4, %t#5, %t#6 : index, index, index, index, index, index, index
}

}

// -----

module attributes {
  hal.device.targets = [
    #hal.device.target<"cpu", {
      buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 16, max_buffer_range = 1073741824, min_buffer_range_alignment = 16>
    }>
  ]
} {

// CHECK-LABEL: @packDynamic
// CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator,
// CHECK-SAME: %[[SIZE_A:.+]]: index, %[[SIZE_B:.+]]: index
func @packDynamic(%allocator: !hal.allocator, %size_a: index, %size_b: index) ->
    (index, index, index, index) {
  %t:4 = hal.allocator.pack<%allocator : !hal.allocator> slices({
    [0, 1] = %size_a,
    [1, 2] = %size_b,
    [2, 3] = %size_a,
  }) : index

  // TODO(#5405): make this math easier to read/track by adding an iree.align
  // instead of expanding to the individual bit twiddling alignment ops.
  // Right now this is too verbose to really test against with anything but
  // a change detector like this.

  // CHECK-DAG: %c0 = constant 0 : index
  // CHECK-DAG: %c-16 = constant -16 : index
  // CHECK-DAG: %c15 = constant 15 : index
  // CHECK-DAG: %0 = addi %arg1, %c15 : index
  // CHECK-DAG: %1 = and %0, %c-16 : index
  // CHECK-DAG: %2 = addi %1, %c15 : index
  // CHECK-DAG: %3 = and %2, %c-16 : index
  // CHECK-DAG: %4 = addi %arg2, %c15 : index
  // CHECK-DAG: %5 = and %4, %c-16 : index
  // CHECK-DAG: %6 = addi %3, %5 : index
  // CHECK-DAG: %7 = addi %6, %c15 : index
  // CHECK-DAG: %8 = and %7, %c-16 : index
  // CHECK-DAG: %9 = addi %8, %c15 : index
  // CHECK-DAG: %10 = and %9, %c-16 : index

  // CHECK-DAG: return %10, %c0, %3, %c0
  return %t#0, %t#1, %t#2, %t#3 : index, index, index, index
}

}

// -----

module attributes {
  hal.device.targets = [
    #hal.device.target<"cpu", {
      buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 16, max_buffer_range = 1073741824, min_buffer_range_alignment = 16>
    }>
  ]
} {

// CHECK-LABEL: @packMixedStaticDynamic
// CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator,
// CHECK-SAME: %[[SIZE_A:.+]]: index, %[[SIZE_B:.+]]: index
func @packMixedStaticDynamic(%allocator: !hal.allocator, %size_a: index, %size_b: index) ->
    (index, index, index, index, index) {
  %c100 = constant 100 : index
  %c200 = constant 200 : index
  %t:5 = hal.allocator.pack<%allocator : !hal.allocator> slices({
    [0, 1] = %c100,
    [1, 2] = %size_a,
    [2, 3] = %size_b,
    [5, 6] = %c200,
  }) : index

  // TODO(#5405): make this math easier to read/track by adding an iree.align
  // instead of expanding to the individual bit twiddling alignment ops.
  // Right now this is too verbose to really test against with anything but
  // a change detector like this.

  // CHECK-DAG: %c0 = constant 0 : index
  // CHECK-DAG: %c208 = constant 208 : index
  // CHECK-DAG: %c-16 = constant -16 : index
  // CHECK-DAG: %c15 = constant 15 : index
  // CHECK-DAG: %0 = addi %arg1, %c15 : index
  // CHECK-DAG: %1 = and %0, %c-16 : index
  // CHECK-DAG: %2 = addi %1, %c223 : index
  // CHECK-DAG: %3 = and %2, %c-16 : index
  // CHECK-DAG: %4 = addi %arg2, %c15 : index
  // CHECK-DAG: %5 = and %4, %c-16 : index
  // CHECK-DAG: %6 = addi %3, %5 : index
  // CHECK-DAG: %7 = addi %6, %c15 : index
  // CHECK-DAG: %8 = and %7, %c-16 : index
  // CHECK-DAG: %9 = addi %8, %c15 : index
  // CHECK-DAG: %10 = and %9, %c-16 : index

  // CHECK-DAG: return %10, %c0, %c208, %3, %c0
  return %t#0, %t#1, %t#2, %t#3, %t#4 : index, index, index, index, index
}

}
