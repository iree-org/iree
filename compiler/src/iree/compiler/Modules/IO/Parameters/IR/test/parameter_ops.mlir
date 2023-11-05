// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @parameterLoad
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence)
func.func @parameterLoad(%device: !hal.device, %wait: !hal.fence, %signal: !hal.fence) {
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  %affinity = arith.constant -1 : i64
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 0
  %offset = arith.constant 0 : i64
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 128
  %length = arith.constant 128 : index
  // CHECK: = io_parameters.load<%[[DEVICE]] : !hal.device>
  // CHECK-SAME: affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]])
  // CHECK-SAME: signal(%[[SIGNAL]])
  // CHECK-SAME: source("scope"::"w0")[%[[OFFSET]]]
  // CHECK-SAME: type("DeviceVisible|DeviceLocal")
  // CHECK-SAME: usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable")
  // CHECK-SAME: : !hal.buffer{%[[LENGTH]]}
  %0 = io_parameters.load<%device : !hal.device>
      affinity(%affinity)
      wait(%wait)
      signal(%signal)
      source("scope"::"w0")[%offset]
      type("DeviceVisible|DeviceLocal")
      usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable")
      : !hal.buffer{%length}
  return
}

// -----
