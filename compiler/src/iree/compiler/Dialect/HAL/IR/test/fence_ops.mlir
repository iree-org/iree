// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @timeline_advance
func.func @timeline_advance() -> !hal.fence {
  // CHECK: = hal.timeline.advance : !hal.fence
  %fence = hal.timeline.advance : !hal.fence
  return %fence : !hal.fence
}

// -----

// CHECK-LABEL: @fence_create
func.func @fence_create(%arg0: !hal.semaphore, %arg1: i64, %arg2: i64) -> !hal.fence {
  // CHECK: = hal.fence.create
  // CHECK-SAME: at<%arg0 : !hal.semaphore>(%arg1)
  // CHECK-SAME: at<%arg0 : !hal.semaphore>(%arg2)
  // CHECK-SAME: -> !hal.fence
  %fence = hal.fence.create
      at<%arg0 : !hal.semaphore>(%arg1)
      at<%arg0 : !hal.semaphore>(%arg2)
      -> !hal.fence
  return %fence : !hal.fence
}

// -----

// CHECK-LABEL: @fence_join
func.func @fence_join(%arg0: !hal.fence, %arg1: !hal.fence) -> !hal.fence {
  // CHECK: = hal.fence.join at([%arg0, %arg1]) -> !hal.fence
  %fence = hal.fence.join at([%arg0, %arg1]) -> !hal.fence
  return %fence : !hal.fence
}

// -----

// CHECK-LABEL: @fence_signal
func.func @fence_signal(%arg0: !hal.fence) {
  // CHECK: hal.fence.signal<%arg0 : !hal.fence>
  hal.fence.signal<%arg0 : !hal.fence>
  return
}

// -----

// CHECK-LABEL: @fence_fail
func.func @fence_fail(%arg0: !hal.fence, %arg1: i32) {
  // CHECK: hal.fence.fail<%arg0 : !hal.fence> status(%arg1)
  hal.fence.fail<%arg0 : !hal.fence> status(%arg1)
  return
}

// -----

// CHECK-LABEL: @fence_await
func.func @fence_await(%arg0: !hal.fence, %arg1: !hal.fence, %arg2: i32) -> i32 {
  // CHECK: = hal.fence.await until([%arg0, %arg1]) timeout_millis(%arg2) : i32
  %status = hal.fence.await until([%arg0, %arg1]) timeout_millis(%arg2) : i32
  return %status : i32
}
