// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: @parameterLoad
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>)
util.func public @parameterLoad(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence) -> (!hal.buffer, !hal.buffer) {
  %param_offset0 = arith.constant 50 : i64
  %param_offset1 = arith.constant 51 : i64
  %size0 = arith.constant 100 : index
  %size1 = arith.constant 101 : index
  %memory_type = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable"> : i32
  %scope = util.buffer.constant : !util.buffer = "scope"
  %key0 = util.buffer.constant : !util.buffer = "key0"
  %key1 = util.buffer.constant : !util.buffer = "key1"
  //  CHECK-DAG: %[[SCOPE:.+]] = vm.rodata.inline {{.+}} = "scope"
  //  CHECK-DAG: %[[KEY_TABLE:.+]], %[[KEY_DATA:.+]] = vm.rodata.table.inline i32 : !vm.buffer, !vm.buffer = ["key0", "key1"]
  //  CHECK-DAG: %[[SPANS:.+]] = vm.rodata.inline : !vm.buffer = dense<[50, 0, 100, 51, 0, 101]> : vector<6xi64>
  //  CHECK-DAG: %[[MEMORY_TYPE:.+]] = vm.const.i32 48
  //  CHECK-DAG: %[[BUFFER_USAGE:.+]] = vm.const.i32 527363
  // CHECK: %[[TARGET_BUFFERS:.+]] = vm.call @io_parameters.load
  // CHECK-SAME: (%[[DEVICE]], %[[QUEUE_AFFINITY]], %[[WAIT]], %[[SIGNAL]],
  // CHECK-SAME:  %[[SCOPE]], %[[QUEUE_AFFINITY]], %[[MEMORY_TYPE]], %[[BUFFER_USAGE]], %[[KEY_TABLE]], %[[KEY_DATA]], %[[SPANS]])
  %target_buffers:2 = io_parameters.load<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) type(%memory_type) usage(%buffer_usage) {
    %scope::%key0[%param_offset0] : !hal.buffer{%size0},
    %scope::%key1[%param_offset1] : !hal.buffer{%size1}
  }
  // CHECK-DAG: %[[C0:.+]] = vm.const.i32 0
  // CHECK-DAG: %[[TARGET_BUFFER0:.+]] = vm.list.get.ref %[[TARGET_BUFFERS]], %[[C0]]
  // CHECK-DAG: %[[C1:.+]] = vm.const.i32 1
  // CHECK-DAG: %[[TARGET_BUFFER1:.+]] = vm.list.get.ref %[[TARGET_BUFFERS]], %[[C1]]
  // CHECK: return %[[TARGET_BUFFER0]], %[[TARGET_BUFFER1]]
  util.return %target_buffers#0, %target_buffers#1 : !hal.buffer, !hal.buffer
}

// -----

// CHECK-LABEL: @parameterLoadNoScope
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>)
util.func public @parameterLoadNoScope(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence) -> !hal.buffer {
  %param_offset = arith.constant 50 : i64
  %size = arith.constant 100 : index
  %memory_type = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable"> : i32
  %key = util.buffer.constant : !util.buffer = "key"
  //  CHECK-DAG: %[[KEY_TABLE:.+]], %[[KEY_DATA:.+]] = vm.rodata.table.inline i32 : !vm.buffer, !vm.buffer = ["key"]
  //  CHECK-DAG: %[[SPANS:.+]] = vm.rodata.inline : !vm.buffer = dense<[50, 0, 100]> : vector<3xi64>
  //  CHECK-DAG: %[[SCOPE:.+]] = vm.const.ref.zero : !vm.buffer
  //  CHECK-DAG: %[[MEMORY_TYPE:.+]] = vm.const.i32 48
  //  CHECK-DAG: %[[BUFFER_USAGE:.+]] = vm.const.i32 527363
  // CHECK: %[[TARGET_BUFFERS:.+]] = vm.call @io_parameters.load
  // CHECK-SAME: (%[[DEVICE]], %[[QUEUE_AFFINITY]], %[[WAIT]], %[[SIGNAL]],
  // CHECK-SAME:  %[[SCOPE]], %[[QUEUE_AFFINITY]], %[[MEMORY_TYPE]], %[[BUFFER_USAGE]], %[[KEY_TABLE]], %[[KEY_DATA]], %[[SPANS]])
  %target_buffer = io_parameters.load<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) type(%memory_type) usage(%buffer_usage) {
    %key[%param_offset] : !hal.buffer{%size}
  }
  // CHECK-DAG: %[[C0:.+]] = vm.const.i32 0
  // CHECK-DAG: %[[TARGET_BUFFER:.+]] = vm.list.get.ref %[[TARGET_BUFFERS]], %[[C0]]
  // CHECK: return %[[TARGET_BUFFER]]
  util.return %target_buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @parameterGather
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>, %[[TARGET_BUFFER:.+]]: !vm.ref<!hal.buffer>)
util.func public @parameterGather(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence, %target_buffer: !hal.buffer) {
  %param_offset0 = arith.constant 50 : i64
  %param_offset1 = arith.constant 51 : i64
  %param_offset2 = arith.constant 52 : i64
  %target_offset0 = arith.constant 100 : index
  %target_offset1 = arith.constant 101 : index
  %target_offset2 = arith.constant 102 : index
  %length0 = arith.constant 200 : index
  %length1 = arith.constant 201 : index
  %length2 = arith.constant 202 : index
  %scope = util.buffer.constant : !util.buffer = "scope"
  %key0 = util.buffer.constant : !util.buffer = "key0"
  %key1 = util.buffer.constant : !util.buffer = "key1"
  %key2 = util.buffer.constant : !util.buffer = "key2"
  //  CHECK-DAG: %[[SCOPE:.+]] = vm.rodata.inline {{.+}} = "scope"
  //  CHECK-DAG: %[[KEY_TABLE:.+]], %[[KEY_DATA:.+]] = vm.rodata.table.inline i32 : !vm.buffer, !vm.buffer = ["key0", "key1", "key2"]
  //  CHECK-DAG: %[[SPANS:.+]] = vm.rodata.inline : !vm.buffer = dense<[50, 100, 200, 51, 101, 201, 52, 102, 202]> : vector<9xi64>
  //      CHECK: vm.call @io_parameters.gather
  // CHECK-SAME: (%[[DEVICE]], %[[QUEUE_AFFINITY]], %[[WAIT]], %[[SIGNAL]],
  // CHECK-SAME:   %[[SCOPE]], %[[TARGET_BUFFER]], %[[KEY_TABLE]], %[[KEY_DATA]], %[[SPANS]])
  io_parameters.gather<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) {
    %scope::%key0[%param_offset0] -> %target_buffer[%target_offset0 for %length0] : !hal.buffer,
    %scope::%key1[%param_offset1] -> %target_buffer[%target_offset1 for %length1] : !hal.buffer,
    %scope::%key2[%param_offset2] -> %target_buffer[%target_offset2 for %length2] : !hal.buffer
  }
  util.return
}

// -----

// CHECK-LABEL: @parameterScatter
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>, %[[SOURCE_BUFFER:.+]]: !vm.ref<!hal.buffer>)
util.func public @parameterScatter(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence, %source_buffer: !hal.buffer) {
  %param_offset0 = arith.constant 50 : i64
  %param_offset1 = arith.constant 51 : i64
  %param_offset2 = arith.constant 52 : i64
  %source_offset0 = arith.constant 100 : index
  %source_offset1 = arith.constant 101 : index
  %source_offset2 = arith.constant 102 : index
  %length0 = arith.constant 200 : index
  %length1 = arith.constant 201 : index
  %length2 = arith.constant 202 : index
  %scope = util.buffer.constant : !util.buffer = "scope"
  %key0 = util.buffer.constant : !util.buffer = "key0"
  %key1 = util.buffer.constant : !util.buffer = "key1"
  %key2 = util.buffer.constant : !util.buffer = "key2"
  //  CHECK-DAG: %[[SCOPE:.+]] = vm.rodata.inline {{.+}} = "scope"
  //  CHECK-DAG: %[[KEY_TABLE:.+]], %[[KEY_DATA:.+]] = vm.rodata.table.inline i32 : !vm.buffer, !vm.buffer = ["key0", "key1", "key2"]
  //  CHECK-DAG: %[[SPANS:.+]] = vm.rodata.inline : !vm.buffer = dense<[50, 100, 200, 51, 101, 201, 52, 102, 202]> : vector<9xi64>
  //      CHECK: vm.call @io_parameters.scatter
  // CHECK-SAME: (%[[DEVICE]], %[[QUEUE_AFFINITY]], %[[WAIT]], %[[SIGNAL]],
  // CHECK-SAME:   %[[SOURCE_BUFFER]], %[[SCOPE]], %[[KEY_TABLE]], %[[KEY_DATA]], %[[SPANS]])
  io_parameters.scatter<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) {
    %source_buffer[%source_offset0 for %length0] : !hal.buffer -> %scope::%key0[%param_offset0],
    %source_buffer[%source_offset1 for %length1] : !hal.buffer -> %scope::%key1[%param_offset1],
    %source_buffer[%source_offset2 for %length2] : !hal.buffer -> %scope::%key2[%param_offset2]
  }
  util.return
}

// -----

// Tests dynamic key table construction for a single dynamic key.
// The key is a function argument (not a vm.rodata.inline), so the conversion
// must construct the key_table and key_data buffers at runtime.

// CHECK-LABEL: @parameterLoadDynamicKey
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>, %[[KEY:.+]]: !vm.buffer)
util.func public @parameterLoadDynamicKey(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence, %key: !util.buffer) -> !hal.buffer {
  %param_offset = arith.constant 0 : i64
  %size = arith.constant 100 : index
  %memory_type = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable"> : i32
  // Dynamic key: construct key_table and key_data at runtime.
  //  CHECK-DAG: %[[KEY_LENGTH:.+]] = vm.buffer.length %[[KEY]]
  //  CHECK-DAG: %[[KEY_DATA:.+]] = vm.buffer.alloc %[[KEY_LENGTH]],
  //  CHECK-DAG: %[[TABLE_SIZE:.+]] = vm.const.i64 8
  //  CHECK-DAG: %[[KEY_TABLE:.+]] = vm.buffer.alloc %[[TABLE_SIZE]],
  //      CHECK: vm.buffer.copy %[[KEY]], %{{.+}}, %[[KEY_DATA]], %{{.+}}, %[[KEY_LENGTH]]
  //      CHECK: vm.buffer.store.i32 %{{.+}}, %[[KEY_TABLE]]
  //      CHECK: vm.buffer.store.i32 %{{.+}}, %[[KEY_TABLE]]
  //      CHECK: vm.call @io_parameters.load
  // CHECK-SAME: %[[KEY_TABLE]], %[[KEY_DATA]]
  %target_buffer = io_parameters.load<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) type(%memory_type) usage(%buffer_usage) {
    %key[%param_offset] : !hal.buffer{%size}
  }
  util.return %target_buffer : !hal.buffer
}

// -----

// Tests dynamic key table construction for multiple dynamic keys.
// Both keys are function arguments, so the conversion must concatenate them
// into a single key_data buffer and build a multi-entry key_table.

// CHECK-LABEL: @parameterGatherDynamicKeys
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>, %[[TARGET_BUFFER:.+]]: !vm.ref<!hal.buffer>, %[[KEY0:.+]]: !vm.buffer, %[[KEY1:.+]]: !vm.buffer)
util.func public @parameterGatherDynamicKeys(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence, %target_buffer: !hal.buffer, %key0: !util.buffer, %key1: !util.buffer) {
  %param_offset0 = arith.constant 0 : i64
  %param_offset1 = arith.constant 0 : i64
  %target_offset0 = arith.constant 0 : index
  %target_offset1 = arith.constant 100 : index
  %length0 = arith.constant 100 : index
  %length1 = arith.constant 100 : index
  %scope = util.buffer.constant : !util.buffer = "scope"
  // Two dynamic keys: concatenate into key_data, build two-entry key_table.
  //  CHECK-DAG: %[[KEY0_LENGTH:.+]] = vm.buffer.length %[[KEY0]]
  //  CHECK-DAG: %[[KEY1_LENGTH:.+]] = vm.buffer.length %[[KEY1]]
  //  CHECK-DAG: %[[TOTAL_DATA:.+]] = vm.add.i64 %[[KEY0_LENGTH]], %[[KEY1_LENGTH]]
  //  CHECK-DAG: %[[KEY_DATA:.+]] = vm.buffer.alloc %[[TOTAL_DATA]],
  //  CHECK-DAG: %[[TABLE_SIZE:.+]] = vm.const.i64 16
  //  CHECK-DAG: %[[KEY_TABLE:.+]] = vm.buffer.alloc %[[TABLE_SIZE]],
  //      CHECK: vm.buffer.copy %[[KEY0]], %{{.+}}, %[[KEY_DATA]]
  //      CHECK: vm.buffer.store.i32 %{{.+}}, %[[KEY_TABLE]]
  //      CHECK: vm.buffer.store.i32 %{{.+}}, %[[KEY_TABLE]]
  //      CHECK: vm.buffer.copy %[[KEY1]], %{{.+}}, %[[KEY_DATA]]
  //      CHECK: vm.buffer.store.i32 %{{.+}}, %[[KEY_TABLE]]
  //      CHECK: vm.buffer.store.i32 %{{.+}}, %[[KEY_TABLE]]
  //      CHECK: vm.call @io_parameters.gather
  // CHECK-SAME: %[[KEY_TABLE]], %[[KEY_DATA]]
  io_parameters.gather<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) {
    %scope::%key0[%param_offset0] -> %target_buffer[%target_offset0 for %length0] : !hal.buffer,
    %scope::%key1[%param_offset1] -> %target_buffer[%target_offset1 for %length1] : !hal.buffer
  }
  util.return
}

// -----

// Tests that a mix of constant and dynamic keys falls back to the dynamic
// key table construction path. Even though the first key is a compile-time
// constant (vm.rodata.inline), the presence of a dynamic second key forces
// all keys through the runtime path.

// CHECK-LABEL: @parameterLoadMixedKeys
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>, %[[DYNAMIC_KEY:.+]]: !vm.buffer)
util.func public @parameterLoadMixedKeys(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence, %dynamic_key: !util.buffer) -> (!hal.buffer, !hal.buffer) {
  %param_offset0 = arith.constant 0 : i64
  %param_offset1 = arith.constant 0 : i64
  %size0 = arith.constant 100 : index
  %size1 = arith.constant 100 : index
  %memory_type = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable"> : i32
  %constant_key = util.buffer.constant : !util.buffer = "const_key"
  // One constant + one dynamic key: dynamic path used for both.
  //      CHECK: %[[CONST_KEY:.+]] = vm.rodata.inline {{.+}} = "const_key"
  //  CHECK-DAG: %[[CK_LENGTH:.+]] = vm.buffer.length %[[CONST_KEY]]
  //  CHECK-DAG: %[[DK_LENGTH:.+]] = vm.buffer.length %[[DYNAMIC_KEY]]
  //  CHECK-DAG: %[[TOTAL_DATA:.+]] = vm.add.i64 %[[CK_LENGTH]], %[[DK_LENGTH]]
  //  CHECK-DAG: %[[KEY_DATA:.+]] = vm.buffer.alloc %[[TOTAL_DATA]],
  //  CHECK-DAG: %[[TABLE_SIZE:.+]] = vm.const.i64 16
  //  CHECK-DAG: %[[KEY_TABLE:.+]] = vm.buffer.alloc %[[TABLE_SIZE]],
  //      CHECK: vm.buffer.copy %[[CONST_KEY]], %{{.+}}, %[[KEY_DATA]]
  //      CHECK: vm.buffer.store.i32 %{{.+}}, %[[KEY_TABLE]]
  //      CHECK: vm.buffer.store.i32 %{{.+}}, %[[KEY_TABLE]]
  //      CHECK: vm.buffer.copy %[[DYNAMIC_KEY]], %{{.+}}, %[[KEY_DATA]]
  //      CHECK: vm.buffer.store.i32 %{{.+}}, %[[KEY_TABLE]]
  //      CHECK: vm.buffer.store.i32 %{{.+}}, %[[KEY_TABLE]]
  //      CHECK: vm.call @io_parameters.load
  // CHECK-SAME: %[[KEY_TABLE]], %[[KEY_DATA]]
  %target_buffers:2 = io_parameters.load<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) type(%memory_type) usage(%buffer_usage) {
    %constant_key[%param_offset0] : !hal.buffer{%size0},
    %dynamic_key[%param_offset1] : !hal.buffer{%size1}
  }
  util.return %target_buffers#0, %target_buffers#1 : !hal.buffer, !hal.buffer
}
