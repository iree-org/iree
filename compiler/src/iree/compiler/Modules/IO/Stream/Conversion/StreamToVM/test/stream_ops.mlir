// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: @console_stdin
func.func @console_stdin() -> !io_stream.handle {
  // CHECK: = vm.call @io_stream.console.stdin() {nosideeffects} : () -> !vm.ref<!io_stream.handle>
  %stdin = io_stream.console.stdin : !io_stream.handle
  return %stdin : !io_stream.handle
}

// -----

// CHECK-LABEL: @queries
// CHECK-SAME: (%[[HANDLE:.+]]: !vm.ref<!io_stream.handle>)
func.func @queries(%handle: !io_stream.handle) {
  // CHECK: = vm.call @io_stream.offset(%[[HANDLE]])
  %offset = io_stream.offset(%handle) : (!io_stream.handle) -> i64
  // CHECK: = vm.call @io_stream.length(%[[HANDLE]])
  %length = io_stream.length(%handle) : (!io_stream.handle) -> i64
  return
}

// -----

// CHECK-LABEL: @read_byte
// CHECK-SAME: (%[[HANDLE:.+]]: !vm.ref<!io_stream.handle>)
func.func @read_byte(%handle: !io_stream.handle) -> (i8, i1) {
  // CHECK: %[[BYTE:.+]] = vm.call @io_stream.read.byte(%[[HANDLE]])
  // CHECK: %[[ZERO:.+]] = vm.const.i32.zero
  // CHECK: %[[EOS:.+]] = vm.cmp.lt.i32.s %[[BYTE]], %[[ZERO]]
  %byte, %eos = io_stream.read.byte(%handle) : (!io_stream.handle) -> (i8, i1)
  // CHECK: vm.return %[[BYTE]], %[[EOS]]
  return %byte, %eos : i8, i1
}

// -----

// CHECK-LABEL: @read_bytes
// CHECK-SAME: (%[[HANDLE:.+]]: !vm.ref<!io_stream.handle>, %[[BUFFER:.+]]: !vm.buffer)
func.func @read_bytes(%handle: !io_stream.handle, %buffer: !util.buffer) {
  // CHECK-DAG: %[[OFFSET:.+]] = vm.const.i64.zero
  // CHECK-DAG: %[[LENGTH:.+]] = vm.buffer.length %[[BUFFER]]
  // CHECK: = vm.call @io_stream.read.bytes(%[[HANDLE]], %[[BUFFER]], %[[OFFSET]], %[[LENGTH]])
  %byte = io_stream.read.bytes(%handle, %buffer) : (!io_stream.handle, !util.buffer) -> index
  return
}

// -----

// CHECK-LABEL: @read_bytes_range
// CHECK-SAME: (%[[HANDLE:.+]]: !vm.ref<!io_stream.handle>, %[[BUFFER:.+]]: !vm.buffer)
func.func @read_bytes_range(%handle: !io_stream.handle, %buffer: !util.buffer) {
  // CHECK-DAG: %[[OFFSET:.+]] = vm.const.i64 100
  %offset = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = vm.const.i64 200
  %length = arith.constant 200 : index
  // CHECK: = vm.call @io_stream.read.bytes(%[[HANDLE]], %[[BUFFER]], %[[OFFSET]], %[[LENGTH]])
  %byte = io_stream.read.bytes(%handle, %buffer, %offset, %length) : (!io_stream.handle, !util.buffer, index, index) -> index
  return
}

// -----

// CHECK-LABEL: @read_delimiter
// CHECK-SAME: (%[[HANDLE:.+]]: !vm.ref<!io_stream.handle>)
func.func @read_delimiter(%handle: !io_stream.handle) {
  // CHECK-DAG: %[[DELIMITER:.+]] = vm.const.i32 13
  %delimiter = arith.constant 13 : i8
  // CHECK: = vm.call @io_stream.read.delimiter(%[[HANDLE]], %[[DELIMITER]])
  %buffer = io_stream.read.delimiter(%handle, %delimiter) : (!io_stream.handle, i8) -> !util.buffer
  return
}

// -----

// CHECK-LABEL: @read_line
// CHECK-SAME: (%[[HANDLE:.+]]: !vm.ref<!io_stream.handle>)
func.func @read_line(%handle: !io_stream.handle) {
  // CHECK-DAG: %[[DELIMITER:.+]] = vm.const.i32 10
  // CHECK: = vm.call @io_stream.read.delimiter(%[[HANDLE]], %[[DELIMITER]])
  %buffer = io_stream.read.line(%handle) : (!io_stream.handle) -> !util.buffer
  return
}

// -----

// CHECK-LABEL: @write_byte
// CHECK-SAME: (%[[HANDLE:.+]]: !vm.ref<!io_stream.handle>)
func.func @write_byte(%handle: !io_stream.handle) {
  // CHECK-DAG: %[[BYTE:.+]] = vm.const.i32 13
  %byte = arith.constant 13 : i8
  // CHECK: vm.call @io_stream.write.byte(%[[HANDLE]], %[[BYTE]])
  io_stream.write.byte(%handle, %byte) : (!io_stream.handle, i8) -> ()
  return
}

// -----

// CHECK-LABEL: @write_newline
// CHECK-SAME: (%[[HANDLE:.+]]: !vm.ref<!io_stream.handle>)
func.func @write_newline(%handle: !io_stream.handle) {
  // CHECK-DAG: %[[NEWLINE:.+]] = vm.const.i32 10
  // CHECK: vm.call @io_stream.write.byte(%[[HANDLE]], %[[NEWLINE]])
  io_stream.write.newline(%handle) : (!io_stream.handle) -> ()
  return
}

// -----

// CHECK-LABEL: @write_bytes
// CHECK-SAME: (%[[HANDLE:.+]]: !vm.ref<!io_stream.handle>, %[[BUFFER:.+]]: !vm.buffer)
func.func @write_bytes(%handle: !io_stream.handle, %buffer: !util.buffer) {
  // CHECK-DAG: %[[OFFSET:.+]] = vm.const.i64.zero
  // CHECK-DAG: %[[LENGTH:.+]] = vm.buffer.length %[[BUFFER]]
  // CHECK: vm.call @io_stream.write.bytes(%[[HANDLE]], %[[BUFFER]], %[[OFFSET]], %[[LENGTH]])
  io_stream.write.bytes(%handle, %buffer) : (!io_stream.handle, !util.buffer) -> ()
  return
}

// -----

// CHECK-LABEL: @write_bytes_range
// CHECK-SAME: (%[[HANDLE:.+]]: !vm.ref<!io_stream.handle>, %[[BUFFER:.+]]: !vm.buffer)
func.func @write_bytes_range(%handle: !io_stream.handle, %buffer: !util.buffer) {
  // CHECK-DAG: %[[OFFSET:.+]] = vm.const.i64 100
  %offset = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = vm.const.i64 200
  %length = arith.constant 200 : index
  // CHECK: vm.call @io_stream.write.bytes(%[[HANDLE]], %[[BUFFER]], %[[OFFSET]], %[[LENGTH]])
  io_stream.write.bytes(%handle, %buffer, %offset, %length) : (!io_stream.handle, !util.buffer, index, index) -> ()
  return
}

// -----

// CHECK-LABEL: @write_line
// CHECK-SAME: (%[[HANDLE:.+]]: !vm.ref<!io_stream.handle>, %[[BUFFER:.+]]: !vm.buffer)
func.func @write_line(%handle: !io_stream.handle, %buffer: !util.buffer) {
  // CHECK-DAG: %[[OFFSET:.+]] = vm.const.i64.zero
  // CHECK-DAG: %[[LENGTH:.+]] = vm.buffer.length %[[BUFFER]]
  // CHECK: vm.call @io_stream.write.bytes(%[[HANDLE]], %[[BUFFER]], %[[OFFSET]], %[[LENGTH]])
  // CHECK-DAG: %[[NEWLINE:.+]] = vm.const.i32 10
  // CHECK: vm.call @io_stream.write.byte(%[[HANDLE]], %[[NEWLINE]])
  io_stream.write.line(%handle, %buffer) : (!io_stream.handle, !util.buffer) -> ()
  return
}
