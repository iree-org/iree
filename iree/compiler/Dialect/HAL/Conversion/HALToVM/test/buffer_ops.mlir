// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: @buffer_subspan
func @buffer_subspan(%arg0 : !hal.buffer) -> !hal.buffer {
  %c42 = arith.constant 42 : index
  %c43 = arith.constant 43 : index
  // CHECK: %ref = vm.call @hal.buffer.subspan(%arg0, %c42, %c43) {nosideeffects} : (!vm.ref<!hal.buffer>, i32, i32) -> !vm.ref<!hal.buffer>
  %buffer = hal.buffer.subspan<%arg0 : !hal.buffer>[%c42, %c43] : !hal.buffer
  return %buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @buffer_load
func @buffer_load(%arg0 : !hal.buffer) -> (i8, i16, i32, f16, f32) {
  %c42 = arith.constant 42 : index
  %c43 = arith.constant 43 : index
  %c44 = arith.constant 44 : index
  %c45 = arith.constant 45 : index
  %c46 = arith.constant 46 : index
  // CHECK: %0 = vm.call @hal.buffer.load(%arg0, %c42, %c1) : (!vm.ref<!hal.buffer>, i32, i32) -> i32
  %0 = hal.buffer.load<%arg0 : !hal.buffer>[%c42] : i8
  // CHECK: %1 = vm.call @hal.buffer.load(%arg0, %c43, %c2) : (!vm.ref<!hal.buffer>, i32, i32) -> i32
  %1 = hal.buffer.load<%arg0 : !hal.buffer>[%c43] : i16
  // CHECK: %2 = vm.call @hal.buffer.load(%arg0, %c44, %c4) : (!vm.ref<!hal.buffer>, i32, i32) -> i32
  %2 = hal.buffer.load<%arg0 : !hal.buffer>[%c44] : i32
  // CHECK: vm.call @hal.buffer.load(%arg0, %c45, %c2_0) : (!vm.ref<!hal.buffer>, i32, i32) -> i32
  %3 = hal.buffer.load<%arg0 : !hal.buffer>[%c45] : f16
  // CHECK: vm.call @hal.buffer.load(%arg0, %c46, %c4_1) : (!vm.ref<!hal.buffer>, i32, i32) -> i32
  %4 = hal.buffer.load<%arg0 : !hal.buffer>[%c46] : f32
  return %0, %1, %2, %3, %4 : i8, i16, i32, f16, f32
}

// -----

// CHECK-LABEL: @buffer_store
func @buffer_store(%arg0 : !hal.buffer, %arg1 : i8, %arg2 : i16, %arg3 : i32, %arg4 : f16, %arg5 : f32) {
  %c42 = arith.constant 42 : index
  %c43 = arith.constant 43 : index
  %c44 = arith.constant 44 : index
  %c45 = arith.constant 45 : index
  %c46 = arith.constant 46 : index
  // CHECK: vm.call @hal.buffer.store(%arg1, %arg0, %c42, %c1) : (i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store<%arg0 : !hal.buffer>[%c42] value(%arg1 : i8)
  // CHECK: vm.call @hal.buffer.store(%arg2, %arg0, %c43, %c2) : (i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store<%arg0 : !hal.buffer>[%c43] value(%arg2 : i16)
  // CHECK: vm.call @hal.buffer.store(%arg3, %arg0, %c44, %c4) : (i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store<%arg0 : !hal.buffer>[%c44] value(%arg3 : i32)
  // CHECK: vm.call @hal.buffer.store(%arg4, %arg0, %c45, %c2_0) : (f32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store<%arg0 : !hal.buffer>[%c45] value(%arg4 : f16)
  // CHECK: vm.call @hal.buffer.store(%arg5, %arg0, %c46, %c4_1) : (f32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store<%arg0 : !hal.buffer>[%c46] value(%arg5 : f32)
  return
}
