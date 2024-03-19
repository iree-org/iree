// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @buffer_constant
util.func public @buffer_constant() -> !util.buffer {
  // CHECK: = util.buffer.constant : !util.buffer = dense<[1, 2, 3]> : tensor<3xi32>
  %0 = util.buffer.constant : !util.buffer = dense<[1, 2, 3]> : tensor<3xi32>
  util.return %0 : !util.buffer
}

// -----

// CHECK-LABEL: @buffer_constant_string
util.func public @buffer_constant_string() -> !util.buffer {
  // CHECK: = util.buffer.constant : !util.buffer = "hello"
  %0 = util.buffer.constant : !util.buffer = "hello"
  util.return %0 : !util.buffer
}

// -----

// CHECK-LABEL: @buffer_alloc
util.func public @buffer_alloc(%arg0: index) -> !util.buffer {
  // CHECK: = util.buffer.alloc uninitialized {alignment = 16 : index} : !util.buffer{%arg0}
  %0 = util.buffer.alloc uninitialized {alignment = 16 : index} : !util.buffer{%arg0}
  util.return %0 : !util.buffer
}

// -----

// CHECK-LABEL: @buffer_dealloc
util.func public @buffer_dealloc(%arg0: !util.buffer, %arg1: index) {
  // CHECK: util.buffer.dealloc %arg0 : !util.buffer{%arg1}
  util.buffer.dealloc %arg0 : !util.buffer{%arg1}
  util.return
}

// -----

// CHECK-LABEL: @buffer_slice
util.func public @buffer_slice(%arg0: !util.buffer, %arg1: index, %arg2: index, %arg3: index) -> !util.buffer {
  // CHECK: = util.buffer.slice %arg0[%arg1] : !util.buffer{%arg2} -> !util.buffer{%arg3}
  %0 = util.buffer.slice %arg0[%arg1] : !util.buffer{%arg2} -> !util.buffer{%arg3}
  util.return %0 : !util.buffer
}

// -----

// CHECK-LABEL: @buffer_subspan
util.func public @buffer_subspan(%arg0: !util.buffer, %arg1: index, %arg2: index, %arg3: index) -> !util.buffer {
  // CHECK: = util.buffer.subspan %arg0[%arg1] : !util.buffer{%arg2} -> !util.buffer{%arg3}
  %0 = util.buffer.subspan %arg0[%arg1] : !util.buffer{%arg2} -> !util.buffer{%arg3}
  util.return %0 : !util.buffer
}

// -----

// CHECK-LABEL: @buffer_size
util.func public @buffer_size(%arg0: !util.buffer) -> index {
  // CHECK: = util.buffer.size %arg0 : !util.buffer
  %0 = util.buffer.size %arg0 : !util.buffer
  util.return %0 : index
}

// -----

// CHECK-LABEL: @buffer_storage
util.func public @buffer_storage(%arg0: !util.buffer, %arg1: index) -> (memref<?xi8>, index) {
  // CHECK: = util.buffer.storage %arg0 : !util.buffer{%arg1} -> (memref<?xi8>, index)
  %0, %1 = util.buffer.storage %arg0 : !util.buffer{%arg1} -> (memref<?xi8>, index)
  util.return %0, %1 : memref<?xi8>, index
}

// -----

// CHECK-LABEL: @buffer_copy
util.func public @buffer_copy(%arg0: !util.buffer, %arg1: index) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: util.buffer.copy %arg0[%c100], %arg0[%c200], %c1 : !util.buffer{%arg1} -> !util.buffer{%arg1}
  util.buffer.copy %arg0[%c100], %arg0[%c200], %c1 : !util.buffer{%arg1} -> !util.buffer{%arg1}
  util.return
}

// -----

// CHECK-LABEL: @buffer_compare
util.func public @buffer_compare(%arg0: !util.buffer, %arg1: index) -> i1 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: = util.buffer.compare %arg0[%c100], %arg0[%c200], %c1 : !util.buffer{%arg1}, !util.buffer{%arg1}
  %0 = util.buffer.compare %arg0[%c100], %arg0[%c200], %c1 : !util.buffer{%arg1}, !util.buffer{%arg1}
  util.return %0 : i1
}

// -----

// CHECK-LABEL: @buffer_fill
util.func public @buffer_fill(%arg0: !util.buffer, %arg1: index, %arg2: i32) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: util.buffer.fill %arg2, %arg0[%c100 for %c200] : i32 -> !util.buffer{%arg1}
  util.buffer.fill %arg2, %arg0[%c100 for %c200] : i32 -> !util.buffer{%arg1}
  util.return
}

// -----

// CHECK-LABEL: @buffer_load
util.func public @buffer_load(%arg0: !util.buffer, %arg1: index) -> i32 {
  %c4 = arith.constant 4 : index
  %c100 = arith.constant 100 : index
  // CHECK: = util.buffer.load %arg0[%c100 for %c4] : !util.buffer{%arg1} -> i32
  %0 = util.buffer.load %arg0[%c100 for %c4] : !util.buffer{%arg1} -> i32
  util.return %0 : i32
}

// -----

// CHECK-LABEL: @buffer_store
util.func public @buffer_store(%arg0: !util.buffer, %arg1: index, %arg2: i32) {
  %c4 = arith.constant 4 : index
  %c100 = arith.constant 100 : index
  // CHECK: util.buffer.store %arg2, %arg0[%c100 for %c4] : i32 -> !util.buffer{%arg1}
  util.buffer.store %arg2, %arg0[%c100 for %c4] : i32 -> !util.buffer{%arg1}
  util.return
}

// -----

// CHECK-LABEL: @buffer_hash
util.func public @buffer_hash(%arg0: !util.buffer, %arg1: index) -> i64 {
  %c17 = arith.constant 17 : index
  %c100 = arith.constant 100 : index
  // CHECK: = util.buffer.hash %arg0[%c100 for %c17] : !util.buffer{%arg1} -> i64
  %0 = util.buffer.hash %arg0[%c100 for %c17] : !util.buffer{%arg1} -> i64
  util.return %0 : i64
}
