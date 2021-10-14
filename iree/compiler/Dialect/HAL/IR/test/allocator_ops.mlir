// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @allocator_compute_size
func @allocator_compute_size(%arg0: !hal.allocator) -> index {
  // CHECK-DAG: %[[DIM0:.+]] = arith.constant 100
  %dim0 = arith.constant 100 : index
  // CHECK-DAG: %[[DIM1:.+]] = arith.constant 200
  %dim1 = arith.constant 200 : index
  // CHECK-DAG: %[[TYPE:.+]] = arith.constant 32
  %type = arith.constant 32 : i32
  // CHECK-DAG: %[[ENCODING:.+]] = arith.constant 1
  %encoding = arith.constant 1 : i32
  // CHECK: %[[SIZE:.+]] = hal.allocator.compute_size<%arg0 : !hal.allocator>
  // CHECK-SAME:                                shape([%[[DIM0]], %[[DIM1]]])
  // CHECK-SAME:                                 type(%[[TYPE]])
  // CHECK-SAME:                             encoding(%[[ENCODING]]) : index
  %sz = hal.allocator.compute_size<%arg0 : !hal.allocator>
                             shape([%dim0, %dim1])
                              type(%type)
                          encoding(%encoding) : index
  // CHECK-NEXT: return %[[SIZE]]
  return %sz : index
}

// -----

// CHECK-LABEL: @allocator_compute_offset
func @allocator_compute_offset(%arg0: !hal.allocator) -> index {
  // CHECK-DAG: %[[IDX0:.+]] = arith.constant 10
  %idx0 = arith.constant 10 : index
  // CHECK-DAG: %[[IDX1:.+]] = arith.constant 20
  %idx1 = arith.constant 20 : index
  // CHECK-DAG: %[[DIM0:.+]] = arith.constant 100
  %dim0 = arith.constant 100 : index
  // CHECK-DAG: %[[DIM1:.+]] = arith.constant 200
  %dim1 = arith.constant 200 : index
  // CHECK-DAG: %[[TYPE:.+]] = arith.constant 32
  %type = arith.constant 32 : i32
  // CHECK-DAG: %[[ENCODING:.+]] = arith.constant 1
  %encoding = arith.constant 1 : i32
  // CHECK: %[[OFFSET:.+]] = hal.allocator.compute_offset<%arg0 : !hal.allocator>
  // CHECK-SAME:                                  indices([%[[IDX0]], %[[IDX1]]])
  // CHECK-SAME:                                    shape([%[[DIM0]], %[[DIM1]]])
  // CHECK-SAME:                                     type(%[[TYPE]])
  // CHECK-SAME:                                 encoding(%[[ENCODING]]) : index
  %off = hal.allocator.compute_offset<%arg0 : !hal.allocator>
                              indices([%idx0, %idx1])
                                shape([%dim0, %dim1])
                                 type(%type)
                             encoding(%encoding) : index
  // CHECK-NEXT: return %[[OFFSET]]
  return %off : index
}

// -----

// CHECK-LABEL: @allocator_compute_range
func @allocator_compute_range(%arg0: !hal.allocator) -> (index, index) {
  // CHECK-DAG: %[[IDX0:.+]] = arith.constant 10
  %idx0 = arith.constant 10 : index
  // CHECK-DAG: %[[IDX1:.+]] = arith.constant 20
  %idx1 = arith.constant 20 : index
  // CHECK-DAG: %[[LEN0:.+]] = arith.constant 11
  %len0 = arith.constant 11 : index
  // CHECK-DAG: %[[LEN1:.+]] = arith.constant 21
  %len1 = arith.constant 21 : index
  // CHECK-DAG: %[[DIM0:.+]] = arith.constant 100
  %dim0 = arith.constant 100 : index
  // CHECK-DAG: %[[DIM1:.+]] = arith.constant 200
  %dim1 = arith.constant 200 : index
  // CHECK-DAG: %[[TYPE:.+]] = arith.constant 32
  %type = arith.constant 32 : i32
  // CHECK-DAG: %[[ENCODING:.+]] = arith.constant 1
  %encoding = arith.constant 1 : i32
  // CHECK: = hal.allocator.compute_range<%arg0 : !hal.allocator>
  // CHECK-SAME:                  indices([%[[IDX0]], %[[IDX1]]])
  // CHECK-SAME:                  lengths([%[[LEN0]], %[[LEN1]]])
  // CHECK-SAME:                    shape([%[[DIM0]], %[[DIM1]]])
  // CHECK-SAME:                     type(%[[TYPE]])
  // CHECK-SAME:                 encoding(%[[ENCODING]]) : index, index
  %off, %len = hal.allocator.compute_range<%arg0 : !hal.allocator>
                                   indices([%idx0, %idx1])
                                   lengths([%len0, %len1])
                                     shape([%dim0, %dim1])
                                      type(%type)
                                  encoding(%encoding) : index, index
  return %off, %len : index, index
}

// -----

// CHECK-LABEL: @allocator_allocate
//  CHECK-SAME: (%[[ALLOCATOR:.+]]: !hal.allocator)
func @allocator_allocate(%allocator: !hal.allocator) {
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant 123
  %size = arith.constant 123 : index
  //      CHECK: %[[REF:.+]] = hal.allocator.allocate<%[[ALLOCATOR]] : !hal.allocator>
  // CHECK-SAME:   type("HostVisible|HostCoherent")
  // CHECK-SAME:   usage(Transfer)
  // CHECK-SAME:   : !hal.buffer{%[[SIZE]]}
  %ref = hal.allocator.allocate<%allocator : !hal.allocator>
      type(HostLocal) usage(Transfer) : !hal.buffer{%size}
  return
}

// -----

// CHECK-LABEL: @allocator_constant_buffer
// CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator
func @allocator_constant_buffer(%allocator: !hal.allocator) {
  //      CHECK: %[[REF:.+]] = hal.allocator.constant<%[[ALLOCATOR]] : !hal.allocator>
  // CHECK-SAME:   type("DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage(Transfer)
  // CHECK-SAME:   : !hal.buffer = dense<123> : tensor<4x4xi32>
  %ref = hal.allocator.constant<%allocator : !hal.allocator>
         type(DeviceLocal) usage(Transfer) : !hal.buffer =
         dense<123> : tensor<4x4xi32>
  return
}

// -----

// CHECK-LABEL: @allocator_map_byte_buffer
//  CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator
func @allocator_map_byte_buffer(%allocator: !hal.allocator, %arg1: !util.byte_buffer) {
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 100
  %offset = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 200
  %length = arith.constant 200 : index
  //      CHECK: = hal.allocator.map<%[[ALLOCATOR]] : !hal.allocator>
  // CHECK-SAME:   source(%arg1 : !util.byte_buffer)[%[[OFFSET]], %[[LENGTH]]]
  // CHECK-SAME:   type("DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage(Transfer)
  // CHECK-SAME:   : !hal.buffer
  %ref = hal.allocator.map<%allocator : !hal.allocator>
                    source(%arg1 : !util.byte_buffer)[%offset, %length]
                    type(DeviceLocal) usage(Transfer) : !hal.buffer
  return
}

// -----

// CHECK-LABEL: @allocator_pack
//  CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator
func @allocator_pack(%allocator: !hal.allocator) {
  // CHECK-DAG: %[[BASE_OFFSET:.+]] = arith.constant 999
  %base_offset = arith.constant 999 : index
  // CHECK-DAG: %[[SIZE_0:.+]] = arith.constant 100
  %size_0 = arith.constant 100 : index
  // CHECK-DAG: %[[SIZE_1:.+]] = arith.constant 101
  %size_1 = arith.constant 101 : index
  // CHECK-DAG: %[[SIZE_2:.+]] = arith.constant 102
  %size_2 = arith.constant 102 : index
  // CHECK-NEXT: %{{.+}}:4 =
  %total_length, %offset_0, %offset_1, %offset_2 =
      // CHECK-SAME: hal.allocator.pack<%[[ALLOCATOR]] : !hal.allocator>
      hal.allocator.pack<%allocator : !hal.allocator>
          // CHECK-SAME: offset(%[[BASE_OFFSET]])
          offset(%base_offset)
          // CHECK-SAME: slices({
          slices({
            // CHECK-NEXT: [0, 4] = %[[SIZE_0]],
            [0, 4] = %size_0,
            // CHECK-NEXT: [1, 2] = %[[SIZE_1]],
            [1, 2] = %size_1,
            // CHECK-NEXT: [3, 4] = %[[SIZE_2]]
            // NOTE: tolerant to trailing comma
            [3, 4] = %size_2,
          // CHECK-NEXT: }) : index
          }) : index
  return
}
