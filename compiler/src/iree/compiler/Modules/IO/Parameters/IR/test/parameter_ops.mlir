// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @parameterLoad
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence)
util.func public @parameterLoad(%device: !hal.device, %wait: !hal.fence, %signal: !hal.fence) {
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  %affinity = arith.constant -1 : i64
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 0
  %offset = arith.constant 0 : i64
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 128
  %length = arith.constant 128 : index
  // CHECK: %[[MEMORY_TYPE:.+]] = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  %memory_type = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  // CHECK: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"{{.+}}Transfer{{.+}}Dispatch{{.+}}"> : i32
  %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable"> : i32
  // CHECK: = io_parameters.load<%[[DEVICE]] : !hal.device>
  // CHECK-SAME: affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]])
  // CHECK-SAME: signal(%[[SIGNAL]])
  // CHECK-SAME: type(%[[MEMORY_TYPE]])
  // CHECK-SAME: usage(%[[BUFFER_USAGE]])
  // CHECK-NEXT: "scope"::"w0"[%[[OFFSET]]] : !hal.buffer{%[[LENGTH]]}
  %0 = io_parameters.load<%device : !hal.device>
      affinity(%affinity)
      wait(%wait)
      signal(%signal)
      type(%memory_type)
      usage(%buffer_usage) {
        "scope"::"w0"[%offset] : !hal.buffer{%length}
      }
  util.return
}
