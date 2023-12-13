// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @file_from_memory
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[BUFFER:.+]]: !util.buffer)
func.func @file_from_memory(%device: !hal.device, %buffer: !util.buffer) -> !hal.file {
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  %affinity = arith.constant -1 : i64
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 100
  %offset = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 200
  %length = arith.constant 200 : index
  // CHECK-DAG: %[[FLAGS:.+]] = arith.constant 0 : i32
  %flags = arith.constant 0 : i32
  // CHECK: = hal.ex.file.from_memory
  // CHECK-SAME: device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME: affinity(%[[AFFINITY]])
  // CHECK-SAME: access(Read)
  // CHECK-SAME: buffer(%[[BUFFER]] : !util.buffer)
  // CHECK-SAME: [%[[OFFSET]] for %[[LENGTH]]]
  // CHECK-SAME: flags(%[[FLAGS]]) : !hal.file
  %file = hal.ex.file.from_memory
      device(%device : !hal.device)
      affinity(%affinity)
      access(Read)
      buffer(%buffer : !util.buffer)[%offset for %length]
      flags(%flags) : !hal.file
  return %file : !hal.file
}
