// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module( util.func(iree-stream-layout-slices, cse))' %s | FileCheck %s

#layoutStaticConfig = #stream.resource_config<{
  max_allocation_size = 1073741824,
  min_buffer_offset_alignment = 16,
  max_buffer_range = 1073741824,
  min_buffer_range_alignment = 16,
  index_bits = 32
}>

// CHECK-LABEL: @layoutStatic
util.func public @layoutStatic() -> (index, index, index, index, index, index, index)
    attributes {stream.resources = #layoutStaticConfig} {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %t:7 = stream.resource.pack slices({
    [0, 1] = %c100,  // +0
    [1, 2] = %c100,  // +112 (100 align 16)
    [2, 3] = %c100,  // +0 (reuse [0, 1])
    [0, 4] = %c200,  // +224 (after 112 + 112; end align 16)
    [5, 6] = %c200,  // +0 (reuse [0, 1]/[2, 3])
    [5, 8] = %c100,  // +208 (after 200 align 16)
  }) : index
  // 224 + 200 align 16 = 432 total bytes required
  // CHECK: util.return %c432
  // CHECK-SAME: %c0, %c112, %c0, %c224, %c0, %c208
  util.return %t#0, %t#1, %t#2, %t#3, %t#4, %t#5, %t#6 : index, index, index, index, index, index, index
}

// -----

#layoutDynamicConfig = #stream.resource_config<{
  max_allocation_size = 1073741824,
  min_buffer_offset_alignment = 16,
  max_buffer_range = 1073741824,
  min_buffer_range_alignment = 16,
  index_bits = 32
}>

// CHECK-LABEL: @layoutDynamic
// CHECK-SAME: (%[[SIZE_A:.+]]: index, %[[SIZE_B:.+]]: index)
util.func public @layoutDynamic(%size_a: index, %size_b: index) -> (index, index, index, index)
    attributes {stream.resources = #layoutDynamicConfig} {
  %t:4 = stream.resource.pack slices({
    [0, 1] = %size_a,
    [1, 2] = %size_b,
    [2, 3] = %size_a,
  }) : index

  // CHECK-DAG: %c0 = arith.constant 0 : index
  // CHECK-DAG: %c16 = arith.constant 16 : index
  // CHECK-DAG: %0 = util.align %[[SIZE_A]], %c16 : index
  // CHECK-DAG: %1 = arith.addi %0, %c0 : index
  // CHECK-DAG: %2 = util.align %[[SIZE_B]], %c16 : index
  // CHECK-DAG: %3 = arith.addi %1, %2 : index

  // CHECK: util.return %3, %c0, %1, %c0
  util.return %t#0, %t#1, %t#2, %t#3 : index, index, index, index
}

// -----

#layoutMixedStaticDynamicConfig = #stream.resource_config<{
  max_allocation_size = 1073741824,
  min_buffer_offset_alignment = 16,
  max_buffer_range = 1073741824,
  min_buffer_range_alignment = 16,
  index_bits = 32
}>

// CHECK-LABEL: @layoutMixedStaticDynamic
// CHECK-SAME: (%[[SIZE_A:.+]]: index, %[[SIZE_B:.+]]: index)
util.func public @layoutMixedStaticDynamic(%size_a: index, %size_b: index) -> (index, index, index, index, index)
    attributes {stream.resources = #layoutMixedStaticDynamicConfig} {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %t:5 = stream.resource.pack slices({
    [0, 1] = %c100,
    [1, 2] = %size_a,
    [2, 3] = %size_b,
    [5, 6] = %c200,
  }) : index

  // CHECK-DAG: %c0 = arith.constant 0 : index
  // CHECK-DAG: %c16 = arith.constant 16 : index
  // CHECK-DAG: %c208 = arith.constant 208 : index
  // CHECK-DAG: %0 = util.align %[[SIZE_A]], %c16 : index
  // CHECK-DAG: %1 = arith.addi %0, %c208 : index
  // CHECK-DAG: %2 = util.align %[[SIZE_B]], %c16 : index
  // CHECK-DAG: %3 = arith.addi %1, %2 : index

  // CHECK: util.return %3, %c0, %c208, %1, %c0
  util.return %t#0, %t#1, %t#2, %t#3, %t#4 : index, index, index, index, index
}
