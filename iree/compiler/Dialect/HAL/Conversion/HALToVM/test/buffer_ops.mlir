// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: @buffer_subspan
func @buffer_subspan(%arg0 : !hal.buffer) -> !hal.buffer {
  %c42 = constant 42 : index
  %c42_0 = constant 42 : index
  // CHECK: %ref = vm.call @hal.buffer.subspan(%arg0, %c42, %c42_0) : (!vm.ref<!hal.buffer>, i32, i32) -> !vm.ref<!hal.buffer>
  %buffer = hal.buffer.subspan %arg0, %c42, %c42_0 : !hal.buffer
  return %buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @buffer_fill
func @buffer_fill(%arg0 : !hal.buffer) {
  %c42 = constant 42 : index
  %c42_0 = constant 42 : index
  %c42_1 = constant 42 : i32
  // CHECK: vm.call @hal.buffer.fill(%arg0, %c42, %c42_0, %c42_1) : (!vm.ref<!hal.buffer>, i32, i32, i32) -> ()
  hal.buffer.fill %arg0, %c42, %c42_0, %c42_1
  return
}

// -----

// CHECK-LABEL: @buffer_read_data
func @buffer_read_data(%arg0 : !hal.buffer, %arg1 : !iree.mutable_byte_buffer) {
  %c42 = constant 42 : index
  %c42_0 = constant 42 : index
  %c42_1 = constant 42 : index
  // CHECK: vm.call @hal.buffer.read_data(%arg0, %c42, %arg1, %c42_0, %c42_1) : (!vm.ref<!hal.buffer>, i32, !vm.ref<!iree.mutable_byte_buffer>, i32, i32) -> ()
  hal.buffer.read_data %arg0, %c42, %arg1, %c42_0, %c42_1 : !iree.mutable_byte_buffer
  return
}

// -----

// CHECK-LABEL: @buffer_write_data
func @buffer_write_data(%arg0 : !hal.buffer, %arg1 : !iree.mutable_byte_buffer) {
  %c42 = constant 42 : index
  %c42_0 = constant 42 : index
  %c42_1 = constant 42 : index
  // CHECK: vm.call @hal.buffer.write_data(%arg1, %c42, %arg0, %c42_0, %c42_1) : (!vm.ref<!iree.mutable_byte_buffer>, i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.write_data %arg1, %c42, %arg0, %c42_0, %c42_1 : !iree.mutable_byte_buffer
  return
}

// -----

// CHECK-LABEL: @buffer_copy_data
func @buffer_copy_data(%arg0 : !hal.buffer, %arg1 : !hal.buffer) {
  %c42 = constant 42 : index
  %c42_0 = constant 42 : index
  %c42_1 = constant 42 : index
  // CHECK: vm.call @hal.buffer.copy_data(%arg0, %c42, %arg1, %c42_0, %c42_1) : (!vm.ref<!hal.buffer>, i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.copy_data %arg0, %c42, %arg1, %c42_0, %c42_1
  return
}

// -----

// CHECK-LABEL: @buffer_load
func @buffer_load(%arg0 : !hal.buffer) -> (i8, i16, i32) {
  %c42 = constant 42 : index
  // CHECK: %0 = vm.call @hal.buffer.load(%arg0, %c42, %c1) : (!vm.ref<!hal.buffer>, i32, i32) -> i32
  %0 = hal.buffer.load %arg0[%c42] : i8
  // CHECK: %1 = vm.call @hal.buffer.load(%arg0, %c42, %c2) : (!vm.ref<!hal.buffer>, i32, i32) -> i32
  %1 = hal.buffer.load %arg0[%c42] : i16
  // CHECK: %2 = vm.call @hal.buffer.load(%arg0, %c42, %c4) : (!vm.ref<!hal.buffer>, i32, i32) -> i32
  %2 = hal.buffer.load %arg0[%c42] : i32
  return %0, %1, %2 : i8, i16, i32
}

// -----

// CHECK-LABEL: @buffer_store
func @buffer_store(%arg0 : !hal.buffer, %arg1 : i8, %arg2 : i16, %arg3 : i32) {
  %c42 = constant 42 : index
  // CHECK: vm.call @hal.buffer.store(%arg1, %arg0, %c42, %c1) : (i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store %arg1, %arg0[%c42] : i8
  // CHECK: vm.call @hal.buffer.store(%arg2, %arg0, %c42, %c2) : (i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store %arg2, %arg0[%c42] : i16
  // CHECK: vm.call @hal.buffer.store(%arg3, %arg0, %c42, %c4) : (i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store %arg3, %arg0[%c42] : i32
  return
}
