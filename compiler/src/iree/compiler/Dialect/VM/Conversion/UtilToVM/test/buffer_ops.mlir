// RUN: iree-opt --split-input-file --iree-vm-conversion --iree-vm-target-index-bits=32 %s | FileCheck %s --check-prefix=CHECK-32
// RUN: iree-opt --split-input-file --iree-vm-conversion --iree-vm-target-index-bits=64 %s | FileCheck %s --check-prefix=CHECK-64

// CHECK-LABEL: @buffer_constant
func.func @buffer_constant() -> !util.buffer {
  // CHECK-64: %[[BUFFER:.+]] = vm.rodata.inline "name"  {alignment = 64 : i64, mime_type = "text/plain"} : !vm.buffer = dense<[1, 2, 3]> : tensor<3xi32>
  %0 = util.buffer.constant "name" {alignment = 64 : index, mime_type = "text/plain"} : !util.buffer = dense<[1, 2, 3]> : tensor<3xi32>
  // CHECK-64: return %[[BUFFER]]
  return %0 : !util.buffer
}

// -----

// CHECK-LABEL: @buffer_constant_string
func.func @buffer_constant_string() -> !util.buffer {
  // CHECK-64: %[[BUFFER:.+]] = vm.rodata.inline : !vm.buffer = "hello"
  %0 = util.buffer.constant : !util.buffer = "hello"
  // CHECK-64: return %[[BUFFER]]
  return %0 : !util.buffer
}

// -----

// CHECK-LABEL: @buffer_alloc
func.func @buffer_alloc(%arg0: index) -> !util.buffer {
  // CHECK-32: %[[SIZE_64:.+]] = vm.ext.i32.i64.u %arg0 : i32 -> i64
  // CHECK-32: %[[BUFFER:.+]] = vm.buffer.alloc %[[SIZE_64]] : !vm.buffer
  // CHECK-64: %[[BUFFER:.+]] = vm.buffer.alloc %arg0 : !vm.buffer
  %0 = util.buffer.alloc uninitialized {alignment = 16 : index} : !util.buffer{%arg0}
  // CHECK-32: return %[[BUFFER]]
  return %0 : !util.buffer
}

// -----

// NOTE: currently not used.

// CHECK-LABEL: @buffer_dealloc
func.func @buffer_dealloc(%arg0: !util.buffer, %arg1: index) {
  // CHECK-32-NOT: util.buffer.dealloc
  util.buffer.dealloc %arg0 : !util.buffer{%arg1}
  return
}

// -----

// CHECK-LABEL: @buffer_slice
func.func @buffer_slice(%arg0: !util.buffer, %arg1: index, %arg2: index, %arg3: index) -> !util.buffer {
  // CHECK-32: %[[SIZE_64:.+]] = vm.ext.i32.i64.u %arg3 : i32 -> i64
  // CHECK-32: %[[BUFFER:.+]] = vm.buffer.alloc %[[SIZE_64]] : !vm.buffer
  // CHECK-32-DAG: %[[ZERO:.+]] = vm.const.i64.zero
  // CHECK-32-DAG: %[[OFFSET_64:.+]] = vm.ext.i32.i64.u %arg1 : i32 -> i64
  // CHECK-32: vm.buffer.copy %arg0, %[[OFFSET_64]], %[[BUFFER]], %[[ZERO]], %[[SIZE_64]] : !vm.buffer -> !vm.buffer
  // CHECK-64-DAG: %[[BUFFER:.+]] = vm.buffer.alloc %arg3 : !vm.buffer
  // CHECK-64-DAG: %[[ZERO:.+]] = vm.const.i64.zero
  // CHECK-64: vm.buffer.copy %arg0, %arg1, %[[BUFFER]], %[[ZERO]], %arg3 : !vm.buffer -> !vm.buffer
  %0 = util.buffer.slice %arg0[%arg1] {alignment = 16 : index} : !util.buffer{%arg2} -> !util.buffer{%arg3}
  // CHECK-32: return %[[BUFFER]]
  return %0 : !util.buffer
}

// -----

// CHECK-LABEL: @buffer_size
func.func @buffer_size(%arg0: !util.buffer) -> index {
  // CHECK-32: %[[SIZE_I64:.+]] = vm.buffer.length %arg0 : !vm.buffer -> i64
  // CHECK-32: %[[SIZE_I32:.+]] = vm.trunc.i64.i32 %[[SIZE_I64]]
  // CHECK-64: %[[SIZE_I64:.+]] = vm.buffer.length %arg0 : !vm.buffer -> i64
  %0 = util.buffer.size %arg0 : !util.buffer
  // CHECK-32: return %[[SIZE_I32]]
  // CHECK-64: return %[[SIZE_I64]]
  return %0 : index
}

// -----

// CHECK-LABEL: @buffer_copy
func.func @buffer_copy(%arg0: !util.buffer, %arg1: index) {
  %c3 = arith.constant 3 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-32-DAG: %[[C3:.+]] = vm.const.i64 3
  // CHECK-32-DAG: %[[C100:.+]] = vm.const.i64 100
  // CHECK-32-DAG: %[[C200:.+]] = vm.const.i64 200
  // CHECK-32: vm.buffer.copy %arg0, %[[C100]], %arg0, %[[C200]], %[[C3]] : !vm.buffer -> !vm.buffer
  // CHECK-64: vm.buffer.copy %arg0, %c100, %arg0, %c200, %c3 : !vm.buffer -> !vm.buffer
  util.buffer.copy %arg0[%c100], %arg0[%c200], %c3 : !util.buffer{%arg1} -> !util.buffer{%arg1}
  return
}

// -----

// CHECK-LABEL: @buffer_compare
func.func @buffer_compare(%arg0: !util.buffer, %arg1: index) -> i1 {
  %c3 = arith.constant 3 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-32-DAG: %[[C3:.+]] = vm.const.i64 3
  // CHECK-32-DAG: %[[C100:.+]] = vm.const.i64 100
  // CHECK-32-DAG: %[[C200:.+]] = vm.const.i64 200
  // CHECK-32: %[[RESULT:.+]] = vm.buffer.compare %arg0, %[[C100]], %arg0, %[[C200]], %[[C3]] : !vm.buffer, !vm.buffer
  // CHECK-64: %[[RESULT:.+]] = vm.buffer.compare %arg0, %c100, %arg0, %c200, %c3 : !vm.buffer, !vm.buffer
  %0 = util.buffer.compare %arg0[%c100], %arg0[%c200], %c3 : !util.buffer{%arg1}, !util.buffer{%arg1}
  // CHECK: return %[[RESULT]]
  return %0 : i1
}

// -----

// CHECK-LABEL: @buffer_fill_i1
func.func @buffer_fill_i1(%arg0: !util.buffer, %arg1: index, %arg2: i1) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-32-DAG: %[[C100:.+]] = vm.const.i64 100
  // CHECK-32-DAG: %[[C200:.+]] = vm.const.i64 200
  // CHECK-32: vm.buffer.fill.i8 %arg0, %[[C100]], %[[C200]], %arg2 : i32 -> !vm.buffer
  // CHECK-64: vm.buffer.fill.i8 %arg0, %c100, %c200, %arg2 : i32 -> !vm.buffer
  util.buffer.fill %arg2, %arg0[%c100 for %c200] : i1 -> !util.buffer{%arg1}
  return
}

// -----

// CHECK-LABEL: @buffer_fill_i8
func.func @buffer_fill_i8(%arg0: !util.buffer, %arg1: index, %arg2: i8) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-32-DAG: %[[C100:.+]] = vm.const.i64 100
  // CHECK-32-DAG: %[[C200:.+]] = vm.const.i64 200
  // CHECK-32: vm.buffer.fill.i8 %arg0, %[[C100]], %[[C200]], %arg2 : i32 -> !vm.buffer
  // CHECK-64: vm.buffer.fill.i8 %arg0, %c100, %c200, %arg2 : i32 -> !vm.buffer
  util.buffer.fill %arg2, %arg0[%c100 for %c200] : i8 -> !util.buffer{%arg1}
  return
}

// -----

// CHECK-LABEL: @buffer_fill_i32
func.func @buffer_fill_i32(%arg0: !util.buffer, %arg1: index, %arg2: i32) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-32-DAG: %[[C25:.+]] = vm.const.i64 25
  // CHECK-32-DAG: %[[C50:.+]] = vm.const.i64 50
  // CHECK-32: vm.buffer.fill.i32 %arg0, %[[C25]], %[[C50]], %arg2 : i32 -> !vm.buffer
  // CHECK-64: vm.buffer.fill.i32 %arg0, %c25, %c50, %arg2 : i32 -> !vm.buffer
  util.buffer.fill %arg2, %arg0[%c100 for %c200] : i32 -> !util.buffer{%arg1}
  return
}

// -----

// CHECK-LABEL: @buffer_fill_i64
func.func @buffer_fill_i64(%arg0: !util.buffer, %arg1: index, %arg2: i64) {
  %c104 = arith.constant 104 : index
  %c208 = arith.constant 208 : index
  // CHECK-32-DAG: %[[C13:.+]] = vm.const.i64 13
  // CHECK-32-DAG: %[[C26:.+]] = vm.const.i64 26
  // CHECK-32: vm.buffer.fill.i64 %arg0, %[[C13]], %[[C26]], %arg2 : i64 -> !vm.buffer
  // CHECK-64: vm.buffer.fill.i64 %arg0, %c13, %c26, %arg2 : i64 -> !vm.buffer
  util.buffer.fill %arg2, %arg0[%c104 for %c208] : i64 -> !util.buffer{%arg1}
  return
}

// -----

// CHECK-LABEL: @buffer_fill_index
func.func @buffer_fill_index(%arg0: !util.buffer, %arg1: index, %arg2: index) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-32: vm.buffer.fill.i32
  // CHECK-64: vm.buffer.fill.i64
  util.buffer.fill %arg2, %arg0[%c100 for %c200] : index -> !util.buffer{%arg1}
  return
}

// -----

// CHECK-LABEL: @buffer_load_i1
func.func @buffer_load_i32(%arg0: !util.buffer, %arg1: index) -> i1 {
  %byte_offset = arith.constant 128 : index
  %element_size = arith.constant 1 : index
  // CHECK-32-DAG: %[[ELEMENT_OFFSET:.+]] = vm.const.i64 128
  // CHECK-32: %[[VALUE:.+]] = vm.buffer.load.i8.s %arg0[%[[ELEMENT_OFFSET]]] : !vm.buffer -> i32
  // CHECK-64-DAG: %[[ELEMENT_OFFSET:.+]] = vm.const.i64 128
  // CHECK-64: %[[VALUE:.+]] = vm.buffer.load.i8.s %arg0[%[[ELEMENT_OFFSET]]] : !vm.buffer -> i32
  %0 = util.buffer.load %arg0[%byte_offset for %element_size] : !util.buffer{%arg1} -> i1
  // CHECK: return %[[VALUE]]
  return %0 : i1
}

// -----

// CHECK-LABEL: @buffer_load_i32
func.func @buffer_load_i32(%arg0: !util.buffer, %arg1: index) -> i32 {
  %byte_offset = arith.constant 128 : index
  %element_size = arith.constant 4 : index
  // CHECK-32-DAG: %[[ELEMENT_OFFSET:.+]] = vm.const.i64 32
  // CHECK-32: %[[VALUE:.+]] = vm.buffer.load.i32 %arg0[%[[ELEMENT_OFFSET]]] : !vm.buffer -> i32
  // CHECK-64-DAG: %[[ELEMENT_OFFSET:.+]] = vm.const.i64 32
  // CHECK-64: %[[VALUE:.+]] = vm.buffer.load.i32 %arg0[%[[ELEMENT_OFFSET]]] : !vm.buffer -> i32
  %0 = util.buffer.load %arg0[%byte_offset for %element_size] : !util.buffer{%arg1} -> i32
  // CHECK: return %[[VALUE]]
  return %0 : i32
}

// -----

// CHECK-LABEL: @buffer_load_i64
func.func @buffer_load_i64(%arg0: !util.buffer, %arg1: index) -> i64 {
  %byte_offset = arith.constant 128 : index
  %element_size = arith.constant 8 : index
  // CHECK-32-DAG: %[[ELEMENT_OFFSET:.+]] = vm.const.i64 16
  // CHECK-32: %[[VALUE:.+]] = vm.buffer.load.i64 %arg0[%[[ELEMENT_OFFSET]]] : !vm.buffer -> i64
  // CHECK-64-DAG: %[[ELEMENT_OFFSET:.+]] = vm.const.i64 16
  // CHECK-64: %[[VALUE:.+]] = vm.buffer.load.i64 %arg0[%[[ELEMENT_OFFSET]]] : !vm.buffer -> i64
  %0 = util.buffer.load %arg0[%byte_offset for %element_size] : !util.buffer{%arg1} -> i64
  // CHECK: return %[[VALUE]]
  return %0 : i64
}

// -----

// CHECK-LABEL: @buffer_load_index
func.func @buffer_load_index(%arg0: !util.buffer, %arg1: index) -> index {
  %byte_offset = arith.constant 100 : index
  %element_size = util.sizeof index
  // CHECK-32: vm.buffer.load.i32
  // CHECK-64: vm.buffer.load.i64
  %0 = util.buffer.load %arg0[%byte_offset for %element_size] : !util.buffer{%arg1} -> index
  return %0 : index
}

// -----

// CHECK-LABEL: @buffer_store_i1
func.func @buffer_store_i1(%arg0: !util.buffer, %arg1: index, %arg2: i1) {
  %byte_offset = arith.constant 128 : index
  %element_size = arith.constant 1 : index
  // CHECK-32-DAG: %[[ELEMENT_OFFSET:.+]] = vm.const.i64 128
  // CHECK-32: vm.buffer.store.i8 %arg2, %arg0[%[[ELEMENT_OFFSET]]] : i32 -> !vm.buffer
  // CHECK-64-DAG: %[[ELEMENT_OFFSET:.+]] = vm.const.i64 128
  // CHECK-64: vm.buffer.store.i8 %arg2, %arg0[%[[ELEMENT_OFFSET]]] : i32 -> !vm.buffer
  util.buffer.store %arg2, %arg0[%byte_offset for %element_size] : i1 -> !util.buffer{%arg1}
  return
}

// -----

// CHECK-LABEL: @buffer_store_i32
func.func @buffer_store_i32(%arg0: !util.buffer, %arg1: index, %arg2: i32) {
  %byte_offset = arith.constant 128 : index
  %element_size = arith.constant 4 : index
  // CHECK-32-DAG: %[[ELEMENT_OFFSET:.+]] = vm.const.i64 32
  // CHECK-32: vm.buffer.store.i32 %arg2, %arg0[%[[ELEMENT_OFFSET]]] : i32 -> !vm.buffer
  // CHECK-64-DAG: %[[ELEMENT_OFFSET:.+]] = vm.const.i64 32
  // CHECK-64: vm.buffer.store.i32 %arg2, %arg0[%[[ELEMENT_OFFSET]]] : i32 -> !vm.buffer
  util.buffer.store %arg2, %arg0[%byte_offset for %element_size] : i32 -> !util.buffer{%arg1}
  return
}

// -----

// CHECK-LABEL: @buffer_store_i64
func.func @buffer_store_i64(%arg0: !util.buffer, %arg1: index, %arg2: i64) {
  %byte_offset = arith.constant 128 : index
  %element_size = arith.constant 8 : index
  // CHECK-32-DAG: %[[ELEMENT_OFFSET:.+]] = vm.const.i64 16
  // CHECK-32: vm.buffer.store.i64 %arg2, %arg0[%[[ELEMENT_OFFSET]]] : i64 -> !vm.buffer
  // CHECK-64-DAG: %[[ELEMENT_OFFSET:.+]] = vm.const.i64 16
  // CHECK-64: vm.buffer.store.i64 %arg2, %arg0[%[[ELEMENT_OFFSET]]] : i64 -> !vm.buffer
  util.buffer.store %arg2, %arg0[%byte_offset for %element_size] : i64 -> !util.buffer{%arg1}
  return
}

// -----

// CHECK-LABEL: @buffer_store_index
func.func @buffer_store_index(%arg0: !util.buffer, %arg1: index, %arg2: index) {
  %byte_offset = arith.constant 100 : index
  %element_size = util.sizeof index
  // CHECK-32: vm.buffer.store.i32
  // CHECK-64: vm.buffer.store.i64
  util.buffer.store %arg2, %arg0[%byte_offset for %element_size] : index -> !util.buffer{%arg1}
  return
}
