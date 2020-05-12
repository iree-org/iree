// Tests printing and parsing of hal.semaphore ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @semaphore_create
func @semaphore_create(%arg0 : !hal.device) -> !hal.semaphore {
  %c0 = std.constant 0 : index
  // CHECK: %semaphore = hal.semaphore.create %arg0, initial_value = %c0 : !hal.semaphore
  %semaphore = hal.semaphore.create %arg0, initial_value = %c0 : !hal.semaphore
  return %semaphore : !hal.semaphore
}

// -----

// CHECK-LABEL: @semaphore_query
func @semaphore_query(%arg0 : !hal.semaphore) {
  // CHECK: = hal.semaphore.query %arg0 : i32, index
  %status, %value = hal.semaphore.query %arg0 : i32, index
  return
}

// -----

// CHECK-LABEL: @semaphore_signal
func @semaphore_signal(%arg0 : !hal.semaphore) {
  %c0 = std.constant 0 : index
  // CHECK: hal.semaphore.signal %arg0, value = %c0
  hal.semaphore.signal %arg0, value = %c0
  return
}

// -----

// CHECK-LABEL: @semaphore_fail
func @semaphore_fail(%arg0 : !hal.semaphore) {
  %c0 = std.constant 0 : i32
  // CHECK: hal.semaphore.fail %arg0, status = %c0
  hal.semaphore.fail %arg0, status = %c0
  return
}

// -----

// CHECK-LABEL: @semaphore_await
func @semaphore_await(%arg0 : !hal.semaphore) {
  %c0 = std.constant 0 : index
  // CHECK: = hal.semaphore.await %arg0, min_value = %c0 : i32
  %0 = hal.semaphore.await %arg0, min_value = %c0 : i32
  return
}
