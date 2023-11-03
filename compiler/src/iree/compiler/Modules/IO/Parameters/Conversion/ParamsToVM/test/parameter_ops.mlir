// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: @parameterLoad
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>)
func.func @parameterLoad(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence) -> !hal.buffer {
  %c50_i64 = arith.constant 50 : i64
  %c100 = arith.constant 100 : index
  // CHECK-DAG: %[[SCOPE:.+]] = vm.rodata.inline {{.+}} = "scope"
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline {{.+}} = "key"
  // CHECK: %[[TARGET_BUFFER:.+]] = vm.call @io_parameters.load
  // CHECK-SAME: (%[[DEVICE]], %[[QUEUE_AFFINITY]], %[[WAIT]], %[[SIGNAL]],
  // CHECK-SAME:  %[[SCOPE]], %[[KEY]], %c50, %c48, %c527363, %c100)
  %target_buffer = io_parameters.load<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) source("scope"::"key")[%c50_i64] type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable") : !hal.buffer{%c100}
  // CHECK: return %[[TARGET_BUFFER]]
  return %target_buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @parameterLoadNoScope
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>)
func.func @parameterLoadNoScope(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence) -> !hal.buffer {
  %c50_i64 = arith.constant 50 : i64
  %c100 = arith.constant 100 : index
  // CHECK-DAG: %[[SCOPE:.+]] = vm.const.ref.zero : !vm.buffer
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline {{.+}} = "key"
  // CHECK: %[[TARGET_BUFFER:.+]] = vm.call @io_parameters.load
  // CHECK-SAME: (%[[DEVICE]], %[[QUEUE_AFFINITY]], %[[WAIT]], %[[SIGNAL]],
  // CHECK-SAME:  %[[SCOPE]], %[[KEY]], %c50, %c48, %c527363, %c100)
  %target_buffer = io_parameters.load<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) source("key")[%c50_i64] type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable") : !hal.buffer{%c100}
  // CHECK: return %[[TARGET_BUFFER]]
  return %target_buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @parameterRead
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>, %[[TARGET_BUFFER:.+]]: !vm.ref<!hal.buffer>)
func.func @parameterRead(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence, %target_buffer: !hal.buffer) {
  %c50_i64 = arith.constant 50 : i64
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-DAG: %[[SCOPE:.+]] = vm.rodata.inline {{.+}} = "scope"
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline {{.+}} = "key"
  // CHECK: vm.call @io_parameters.read
  // CHECK-SAME: (%[[DEVICE]], %[[QUEUE_AFFINITY]], %[[WAIT]], %[[SIGNAL]],
  // CHECK-SAME:  %[[SCOPE]], %[[KEY]], %c50, %[[TARGET_BUFFER]], %c100, %c200)
  io_parameters.read<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) source("scope"::"key")[%c50_i64] target(%target_buffer : !hal.buffer)[%c100] length(%c200)
  return
}

// -----

// CHECK-LABEL: @parameterWrite
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[QUEUE_AFFINITY:.+]]: i64, %[[WAIT:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL:.+]]: !vm.ref<!hal.fence>, %[[SOURCE_BUFFER:.+]]: !vm.ref<!hal.buffer>)
func.func @parameterWrite(%device: !hal.device, %queue_affinity: i64, %wait: !hal.fence, %signal: !hal.fence, %source_buffer: !hal.buffer) {
  %c50_i64 = arith.constant 50 : i64
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-DAG: %[[SCOPE:.+]] = vm.rodata.inline {{.+}} = "scope"
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline {{.+}} = "key"
  // CHECK: vm.call @io_parameters.write
  // CHECK-SAME: (%[[DEVICE]], %[[QUEUE_AFFINITY]], %[[WAIT]], %[[SIGNAL]],
  // CHECK-SAME:  %[[SCOPE]], %[[KEY]], %c50, %[[SOURCE_BUFFER]], %c100, %c200
  io_parameters.write<%device : !hal.device> affinity(%queue_affinity) wait(%wait) signal(%signal) source(%source_buffer : !hal.buffer)[%c100] target("scope"::"key")[%c50_i64] length(%c200)
  return
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
  //  CHECK-DAG: %[[KEY_TABLE:.+]] = vm.rodata.inline : !vm.buffer = dense<[0, 4, 4, 4, 8, 4]> : vector<6xi32>
  //  CHECK-DAG: %[[KEY_DATA:.+]] = vm.rodata.inline : !vm.buffer = #util.composite<12xi8, [
  // CHECK-NEXT:  "key0",
  // CHECK-NEXT:  "key1",
  // CHECK-NEXT:  "key2",
  // CHECK-NEXT: ]>
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
  //  CHECK-DAG: %[[KEY_TABLE:.+]] = vm.rodata.inline : !vm.buffer = dense<[0, 4, 4, 4, 8, 4]> : vector<6xi32>
  //  CHECK-DAG: %[[KEY_DATA:.+]] = vm.rodata.inline : !vm.buffer = #util.composite<12xi8, [
  // CHECK-NEXT:  "key0",
  // CHECK-NEXT:  "key1",
  // CHECK-NEXT:  "key2",
  // CHECK-NEXT: ]>
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
