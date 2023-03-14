// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @fence_create
func.func @fence_create(%arg0: !hal.device) -> !hal.fence {
  // CHECK: = hal.fence.create device(%arg0 : !hal.device) flags("None") : !hal.fence
  %fence = hal.fence.create device(%arg0 : !hal.device) flags("None") : !hal.fence
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

// CHECK-LABEL: @fence_query
func.func @fence_query(%arg0: !hal.fence) -> i32 {
  // CHECK: = hal.fence.query<%arg0 : !hal.fence> : i32
  %status = hal.fence.query<%arg0 : !hal.fence> : i32
  return %status : i32
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
