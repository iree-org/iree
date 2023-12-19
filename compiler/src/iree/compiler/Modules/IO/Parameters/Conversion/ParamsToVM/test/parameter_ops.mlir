// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: @parameterLoad
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>)
func.func @parameterLoad(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence) -> (!hal.buffer, !hal.buffer) {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  //      CHECK: %[[KEY_TABLE:.+]], %[[KEY_DATA:.+]] = vm.rodata.table.inline i32 : !vm.buffer, !vm.buffer = ["key0", "key1"]
  //  CHECK-DAG: %[[SPANS:.+]] = vm.rodata.inline : !vm.buffer = dense<[50, 0, 100, 51, 0, 101]> : vector<6xi64>
  //  CHECK-DAG: %[[SCOPE:.+]] = vm.rodata.inline {{.+}} = "scope"
  // CHECK: %[[TARGET_BUFFERS:.+]] = vm.call @io_parameters.load
  // CHECK-SAME: (%[[DEVICE]], %[[QUEUE_AFFINITY]], %[[WAIT]], %[[SIGNAL]],
  // CHECK-SAME:  %[[SCOPE]], %[[QUEUE_AFFINITY]], %c48, %c527363, %[[KEY_TABLE]], %[[KEY_DATA]], %[[SPANS]])
  %target_buffers:2 = io_parameters.load<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable") {
    "scope"::"key0"[%c50_i64] : !hal.buffer{%c100},
    "scope"::"key1"[%c51_i64] : !hal.buffer{%c101}
  }
  // CHECK-DAG: %[[C0:.+]] = vm.const.i32 0
  // CHECK-DAG: %[[TARGET_BUFFER0:.+]] = vm.list.get.ref %[[TARGET_BUFFERS]], %[[C0]]
  // CHECK-DAG: %[[C1:.+]] = vm.const.i32 1
  // CHECK-DAG: %[[TARGET_BUFFER1:.+]] = vm.list.get.ref %[[TARGET_BUFFERS]], %[[C1]]
  // CHECK: return %[[TARGET_BUFFER0]], %[[TARGET_BUFFER1]]
  return %target_buffers#0, %target_buffers#1 : !hal.buffer, !hal.buffer
}

// -----

// CHECK-LABEL: @parameterLoadNoScope
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>)
func.func @parameterLoadNoScope(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence) -> !hal.buffer {
  %c50_i64 = arith.constant 50 : i64
  %c100 = arith.constant 100 : index
  //      CHECK: %[[KEY_TABLE:.+]], %[[KEY_DATA:.+]] = vm.rodata.table.inline i32 : !vm.buffer, !vm.buffer = ["key"]
  //  CHECK-DAG: %[[SPANS:.+]] = vm.rodata.inline : !vm.buffer = dense<[50, 0, 100]> : vector<3xi64>
  //  CHECK-DAG: %[[SCOPE:.+]] = vm.const.ref.zero : !vm.buffer
  // CHECK: %[[TARGET_BUFFERS:.+]] = vm.call @io_parameters.load
  // CHECK-SAME: (%[[DEVICE]], %[[QUEUE_AFFINITY]], %[[WAIT]], %[[SIGNAL]],
  // CHECK-SAME:  %[[SCOPE]], %[[QUEUE_AFFINITY]], %c48, %c527363, %[[KEY_TABLE]], %[[KEY_DATA]], %[[SPANS]])
  %target_buffer = io_parameters.load<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable") {
    "key"[%c50_i64] : !hal.buffer{%c100}
  }
  // CHECK-DAG: %[[C0:.+]] = vm.const.i32 0
  // CHECK-DAG: %[[TARGET_BUFFER:.+]] = vm.list.get.ref %[[TARGET_BUFFERS]], %[[C0]]
  // CHECK: return %[[TARGET_BUFFER]]
  return %target_buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @parameterGather
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>, %[[TARGET_BUFFER:.+]]: !vm.ref<!hal.buffer>)
func.func @parameterGather(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence, %target_buffer: !hal.buffer) {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c52_i64 = arith.constant 52 : i64
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %c102 = arith.constant 102 : index
  %c200 = arith.constant 200 : index
  %c201 = arith.constant 201 : index
  %c202 = arith.constant 202 : index
  //      CHECK: %[[KEY_TABLE:.+]], %[[KEY_DATA:.+]] = vm.rodata.table.inline i32 : !vm.buffer, !vm.buffer = ["key0", "key1", "key2"]
  //  CHECK-DAG: %[[SPANS:.+]] = vm.rodata.inline : !vm.buffer = dense<[50, 100, 200, 51, 101, 201, 52, 102, 202]> : vector<9xi64>
  //  CHECK-DAG: %[[SCOPE:.+]] = vm.rodata.inline {{.+}} = "scope"
  //      CHECK: vm.call @io_parameters.gather
  // CHECK-SAME: (%[[DEVICE]], %[[QUEUE_AFFINITY]], %[[WAIT]], %[[SIGNAL]],
  // CHECK-SAME:   %[[SCOPE]], %[[TARGET_BUFFER]], %[[KEY_TABLE]], %[[KEY_DATA]], %[[SPANS]])
  io_parameters.gather<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) {
    "scope"::"key0"[%c50_i64] -> %target_buffer[%c100 for %c200] : !hal.buffer,
    "scope"::"key1"[%c51_i64] -> %target_buffer[%c101 for %c201] : !hal.buffer,
    "scope"::"key2"[%c52_i64] -> %target_buffer[%c102 for %c202] : !hal.buffer
  }
  return
}

// -----

// CHECK-LABEL: @parameterScatter
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>, %[[SOURCE_BUFFER:.+]]: !vm.ref<!hal.buffer>)
func.func @parameterScatter(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence, %source_buffer: !hal.buffer) {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c52_i64 = arith.constant 52 : i64
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %c102 = arith.constant 102 : index
  %c200 = arith.constant 200 : index
  %c201 = arith.constant 201 : index
  %c202 = arith.constant 202 : index
  //      CHECK: %[[KEY_TABLE:.+]], %[[KEY_DATA:.+]] = vm.rodata.table.inline i32 : !vm.buffer, !vm.buffer = ["key0", "key1", "key2"]
  //  CHECK-DAG: %[[SPANS:.+]] = vm.rodata.inline : !vm.buffer = dense<[50, 100, 200, 51, 101, 201, 52, 102, 202]> : vector<9xi64>
  //  CHECK-DAG: %[[SCOPE:.+]] = vm.rodata.inline {{.+}} = "scope"
  //      CHECK: vm.call @io_parameters.scatter
  // CHECK-SAME: (%[[DEVICE]], %[[QUEUE_AFFINITY]], %[[WAIT]], %[[SIGNAL]],
  // CHECK-SAME:   %[[SCOPE]], %[[SOURCE_BUFFER]], %[[KEY_TABLE]], %[[KEY_DATA]], %[[SPANS]])
  io_parameters.scatter<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) {
    %source_buffer[%c100 for %c200] : !hal.buffer -> "scope"::"key0"[%c50_i64],
    %source_buffer[%c101 for %c201] : !hal.buffer -> "scope"::"key1"[%c51_i64],
    %source_buffer[%c102 for %c202] : !hal.buffer -> "scope"::"key2"[%c52_i64]
  }
  return
}
