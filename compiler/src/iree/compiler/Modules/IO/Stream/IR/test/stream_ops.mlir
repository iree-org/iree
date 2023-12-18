// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @console_stdin
func.func @console_stdin() -> !io_stream.handle {
  // CHECK: = io_stream.console.stdin : !io_stream.handle
  %stdin = io_stream.console.stdin : !io_stream.handle
  return %stdin : !io_stream.handle
}

// -----

// CHECK-LABEL: @queries
// CHECK-SAME: (%[[HANDLE:.+]]: !io_stream.handle)
func.func @queries(%handle: !io_stream.handle) {
  // CHECK: = io_stream.offset(%[[HANDLE]]) : (!io_stream.handle) -> i64
  %offset = io_stream.offset(%handle) : (!io_stream.handle) -> i64
  // CHECK: = io_stream.length(%[[HANDLE]]) : (!io_stream.handle) -> i64
  %length = io_stream.length(%handle) : (!io_stream.handle) -> i64
  return
}

// -----

// CHECK-LABEL: @read_byte
// CHECK-SAME: (%[[HANDLE:.+]]: !io_stream.handle)
func.func @read_byte(%handle: !io_stream.handle) {
  // CHECK: = io_stream.read.byte(%[[HANDLE]]) : (!io_stream.handle) -> (i8, i1)
  %byte, %eos = io_stream.read.byte(%handle) : (!io_stream.handle) -> (i8, i1)
  return
}

// -----

// CHECK-LABEL: @read_bytes
// CHECK-SAME: (%[[HANDLE:.+]]: !io_stream.handle, %[[BUFFER:.+]]: !util.buffer)
func.func @read_bytes(%handle: !io_stream.handle, %buffer: !util.buffer) {
  // CHECK: = io_stream.read.bytes(%[[HANDLE]], %[[BUFFER]]) : (!io_stream.handle, !util.buffer) -> index
  %byte = io_stream.read.bytes(%handle, %buffer) : (!io_stream.handle, !util.buffer) -> index
  return
}

// -----

// CHECK-LABEL: @read_bytes_range
// CHECK-SAME: (%[[HANDLE:.+]]: !io_stream.handle, %[[BUFFER:.+]]: !util.buffer)
func.func @read_bytes_range(%handle: !io_stream.handle, %buffer: !util.buffer) {
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 100 : index
  %offset = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 200 : index
  %length = arith.constant 200 : index
  // CHECK: = io_stream.read.bytes(%[[HANDLE]], %[[BUFFER]], %[[OFFSET]], %[[LENGTH]]) : (!io_stream.handle, !util.buffer, index, index) -> index
  %byte = io_stream.read.bytes(%handle, %buffer, %offset, %length) : (!io_stream.handle, !util.buffer, index, index) -> index
  return
}

// -----

// CHECK-LABEL: @read_delimiter
// CHECK-SAME: (%[[HANDLE:.+]]: !io_stream.handle)
func.func @read_delimiter(%handle: !io_stream.handle) {
  // CHECK-DAG: %[[DELIMITER:.+]] = arith.constant 13 : i8
  %delimiter = arith.constant 13 : i8
  // CHECK: = io_stream.read.delimiter(%[[HANDLE]], %[[DELIMITER]]) : (!io_stream.handle, i8) -> !util.buffer
  %buffer = io_stream.read.delimiter(%handle, %delimiter) : (!io_stream.handle, i8) -> !util.buffer
  return
}

// -----

// CHECK-LABEL: @read_line
// CHECK-SAME: (%[[HANDLE:.+]]: !io_stream.handle)
func.func @read_line(%handle: !io_stream.handle) {
  // CHECK: = io_stream.read.line(%[[HANDLE]]) : (!io_stream.handle) -> !util.buffer
  %buffer = io_stream.read.line(%handle) : (!io_stream.handle) -> !util.buffer
  return
}

// -----

// CHECK-LABEL: @write_byte
// CHECK-SAME: (%[[HANDLE:.+]]: !io_stream.handle)
func.func @write_byte(%handle: !io_stream.handle) {
  // CHECK-DAG: %[[BYTE:.+]] = arith.constant 13 : i8
  %byte = arith.constant 13 : i8
  // CHECK: io_stream.write.byte(%[[HANDLE]], %[[BYTE]]) : (!io_stream.handle, i8) -> ()
  io_stream.write.byte(%handle, %byte) : (!io_stream.handle, i8) -> ()
  return
}

// -----

// CHECK-LABEL: @write_newline
// CHECK-SAME: (%[[HANDLE:.+]]: !io_stream.handle)
func.func @write_newline(%handle: !io_stream.handle) {
  // CHECK: io_stream.write.newline(%[[HANDLE]]) : (!io_stream.handle) -> ()
  io_stream.write.newline(%handle) : (!io_stream.handle) -> ()
  return
}

// -----

// CHECK-LABEL: @write_bytes
// CHECK-SAME: (%[[HANDLE:.+]]: !io_stream.handle, %[[BUFFER:.+]]: !util.buffer)
func.func @write_bytes(%handle: !io_stream.handle, %buffer: !util.buffer) {
  // CHECK: io_stream.write.bytes(%[[HANDLE]], %[[BUFFER]]) : (!io_stream.handle, !util.buffer) -> ()
  io_stream.write.bytes(%handle, %buffer) : (!io_stream.handle, !util.buffer) -> ()
  return
}

// -----

// CHECK-LABEL: @write_bytes_range
// CHECK-SAME: (%[[HANDLE:.+]]: !io_stream.handle, %[[BUFFER:.+]]: !util.buffer)
func.func @write_bytes_range(%handle: !io_stream.handle, %buffer: !util.buffer) {
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 100 : index
  %offset = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 200 : index
  %length = arith.constant 200 : index
  // CHECK: io_stream.write.bytes(%[[HANDLE]], %[[BUFFER]], %[[OFFSET]], %[[LENGTH]]) : (!io_stream.handle, !util.buffer, index, index) -> ()
  io_stream.write.bytes(%handle, %buffer, %offset, %length) : (!io_stream.handle, !util.buffer, index, index) -> ()
  return
}

// -----

// CHECK-LABEL: @write_line
// CHECK-SAME: (%[[HANDLE:.+]]: !io_stream.handle, %[[BUFFER:.+]]: !util.buffer)
func.func @write_line(%handle: !io_stream.handle, %buffer: !util.buffer) {
  // CHECK: io_stream.write.line(%[[HANDLE]], %[[BUFFER]]) : (!io_stream.handle, !util.buffer) -> ()
  io_stream.write.line(%handle, %buffer) : (!io_stream.handle, !util.buffer) -> ()
  return
}
