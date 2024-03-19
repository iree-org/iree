// RUN: iree-opt --split-input-file --iree-convert-hal-to-vm --cse --iree-vm-target-index-bits=32 %s | FileCheck %s

// CHECK-LABEL: @buffer_subspan
// CHECK-SAME: (%[[BUFFER:.+]]: !vm.ref<!hal.buffer>)
util.func public @buffer_subspan(%buffer : !hal.buffer) -> !hal.buffer {
  %c42 = arith.constant 42 : index
  %c43 = arith.constant 43 : index
  // CHECK: %[[RET:.+]] = vm.call @hal.buffer.subspan(%[[BUFFER]], %c42, %c43) {nosideeffects} : (!vm.ref<!hal.buffer>, i64, i64) -> !vm.ref<!hal.buffer>
  %subspan = hal.buffer.subspan<%buffer : !hal.buffer>[%c42, %c43] : !hal.buffer
  // CHECK: vm.return %[[RET]]
  util.return %subspan: !hal.buffer
}

// -----

// CHECK-LABEL: @buffer_load_i8
// CHECK-SAME: (%[[BUFFER:.+]]: !vm.ref<!hal.buffer>)
util.func public @buffer_load_i8(%buffer: !hal.buffer) -> i8 {
  %c64 = arith.constant 64 : index
  // CHECK: %[[RET:.+]] = vm.call @hal.buffer.load(%[[BUFFER]], %c64, %c1) : (!vm.ref<!hal.buffer>, i64, i32) -> i32
  %0 = hal.buffer.load<%buffer: !hal.buffer>[%c64] : i8
  // CHECK: vm.return %[[RET]]
  util.return %0 : i8
}

// -----

// CHECK-LABEL: @buffer_load_i16
// CHECK-SAME: (%[[BUFFER:.+]]: !vm.ref<!hal.buffer>)
util.func public @buffer_load_i16(%buffer: !hal.buffer) -> i16 {
  %c64 = arith.constant 64 : index
  // CHECK: %[[RET:.+]] = vm.call @hal.buffer.load(%[[BUFFER]], %c64, %c2) : (!vm.ref<!hal.buffer>, i64, i32) -> i32
  %0 = hal.buffer.load<%buffer: !hal.buffer>[%c64] : i16
  // CHECK: vm.return %[[RET]]
  util.return %0 : i16
}

// -----

// CHECK-LABEL: @buffer_load_i32
// CHECK-SAME: (%[[BUFFER:.+]]: !vm.ref<!hal.buffer>)
util.func public @buffer_load_i32(%buffer: !hal.buffer) -> i32 {
  %c64 = arith.constant 64 : index
  // CHECK: %[[RET:.+]] = vm.call @hal.buffer.load(%[[BUFFER]], %c64, %c4) : (!vm.ref<!hal.buffer>, i64, i32) -> i32
  %0 = hal.buffer.load<%buffer: !hal.buffer>[%c64] : i32
  // CHECK: vm.return %[[RET]]
  util.return %0 : i32
}

// -----

// CHECK-LABEL: @buffer_load_i64
// CHECK-SAME: (%[[BUFFER:.+]]: !vm.ref<!hal.buffer>)
util.func public @buffer_load_i64(%buffer: !hal.buffer) -> i64 {
  %c64 = arith.constant 64 : index

  // CHECK-DAG: %[[OFFSET_HI:.+]] = vm.add.i64 %c64, %c4
  // CHECK-DAG: %[[HI_I32:.+]] = vm.call @hal.buffer.load(%[[BUFFER]], %[[OFFSET_HI]], %c4) : (!vm.ref<!hal.buffer>, i64, i32) -> i32
  // CHECK-DAG: %[[HI_EXT:.+]] = vm.ext.i32.i64.u %[[HI_I32]]
  // CHECK-DAG: %[[HI_I64:.+]] = vm.shl.i64 %[[HI_EXT]], %c32

  // CHECK-DAG: %[[LO_I32:.+]] = vm.call @hal.buffer.load(%[[BUFFER]], %c64, %c4) : (!vm.ref<!hal.buffer>, i64, i32) -> i32
  // CHECK-DAG: %[[LO_I64:.+]] = vm.ext.i32.i64.u %[[LO_I32]]

  // CHECK: %[[RET:.+]] = vm.or.i64 %[[LO_I64]], %[[HI_I64]]

  %0 = hal.buffer.load<%buffer: !hal.buffer>[%c64] : i64
  // CHECK: vm.return %[[RET]]
  util.return %0 : i64
}

// -----

// CHECK-LABEL: @buffer_load_f32
// CHECK-SAME: (%[[BUFFER:.+]]: !vm.ref<!hal.buffer>)
util.func public @buffer_load_f32(%buffer: !hal.buffer) -> f32 {
  %c64 = arith.constant 64 : index
  // CHECK: %[[RET_I32:.+]] = vm.call @hal.buffer.load(%[[BUFFER]], %c64, %c4) : (!vm.ref<!hal.buffer>, i64, i32) -> i32
  %0 = hal.buffer.load<%buffer: !hal.buffer>[%c64] : f32
  // CHECK: %[[RET:.+]] = vm.bitcast.i32.f32 %[[RET_I32]]
  // CHECK: vm.return %[[RET]]
  util.return %0 : f32
}

// -----

// CHECK-LABEL: @buffer_store_i8
// CHECK-SAME: (%[[BUFFER:.+]]: !vm.ref<!hal.buffer>, %[[VALUE:.+]]: i32)
util.func public @buffer_store_i8(%buffer: !hal.buffer, %value: i8) {
  %c64 = arith.constant 64 : index
  // CHECK: vm.call @hal.buffer.store(%[[VALUE]], %[[BUFFER]], %c64, %c1) : (i32, !vm.ref<!hal.buffer>, i64, i32) -> ()
  hal.buffer.store<%buffer : !hal.buffer>[%c64] value(%value : i8)
  util.return
}

// -----

// CHECK-LABEL: @buffer_store_i16
// CHECK-SAME: (%[[BUFFER:.+]]: !vm.ref<!hal.buffer>, %[[VALUE:.+]]: i32)
util.func public @buffer_store_i16(%buffer: !hal.buffer, %value: i16) {
  %c64 = arith.constant 64 : index
  // CHECK: vm.call @hal.buffer.store(%[[VALUE]], %[[BUFFER]], %c64, %c2) : (i32, !vm.ref<!hal.buffer>, i64, i32) -> ()
  hal.buffer.store<%buffer : !hal.buffer>[%c64] value(%value : i16)
  util.return
}

// -----

// CHECK-LABEL: @buffer_store_i32
// CHECK-SAME: (%[[BUFFER:.+]]: !vm.ref<!hal.buffer>, %[[VALUE:.+]]: i32)
util.func public @buffer_store_i32(%buffer: !hal.buffer, %value: i32) {
  %c64 = arith.constant 64 : index
  // CHECK: vm.call @hal.buffer.store(%[[VALUE]], %[[BUFFER]], %c64, %c4) : (i32, !vm.ref<!hal.buffer>, i64, i32) -> ()
  hal.buffer.store<%buffer : !hal.buffer>[%c64] value(%value : i32)
  util.return
}

// -----

// CHECK-LABEL: @buffer_store_i64
// CHECK-SAME: (%[[BUFFER:.+]]: !vm.ref<!hal.buffer>, %[[VALUE:.+]]: i64)
util.func public @buffer_store_i64(%buffer: !hal.buffer, %value: i64) {
  %c64 = arith.constant 64 : index

  // CHECK-DAG: %[[VALUE_LO:.+]] = vm.trunc.i64.i32 %[[VALUE]]
  // CHECK: vm.call @hal.buffer.store(%[[VALUE_LO]], %[[BUFFER]], %c64, %c4) : (i32, !vm.ref<!hal.buffer>, i64, i32) -> ()

  // CHECK-DAG: %[[VALUE_HI_I64:.+]] = vm.shr.i64.u %[[VALUE]], %c32
  // CHECK-DAG: %[[VALUE_HI:.+]] = vm.trunc.i64.i32 %[[VALUE_HI_I64]]
  // CHECK-DAG: %[[OFFSET_HI:.+]] = vm.add.i64 %c64, %c4
  // CHECK: vm.call @hal.buffer.store(%[[VALUE_HI]], %[[BUFFER]], %[[OFFSET_HI]], %c4) : (i32, !vm.ref<!hal.buffer>, i64, i32) -> ()

  hal.buffer.store<%buffer : !hal.buffer>[%c64] value(%value : i64)
  util.return
}

// -----

// CHECK-LABEL: @buffer_store_f32
// CHECK-SAME: (%[[BUFFER:.+]]: !vm.ref<!hal.buffer>, %[[VALUE:.+]]: f32)
util.func public @buffer_store_f32(%buffer: !hal.buffer, %value: f32) {
  %c64 = arith.constant 64 : index
  // CHECK: %[[VALUE_I32:.+]] = vm.bitcast.f32.i32 %[[VALUE]]
  // CHECK: vm.call @hal.buffer.store(%[[VALUE_I32]], %[[BUFFER]], %c64, %c4) : (i32, !vm.ref<!hal.buffer>, i64, i32) -> ()
  hal.buffer.store<%buffer : !hal.buffer>[%c64] value(%value : f32)
  util.return
}
