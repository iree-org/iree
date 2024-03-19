// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @fence_create
util.func public @fence_create(%arg0: !hal.device) -> !hal.fence {
  // CHECK: = hal.fence.create device(%arg0 : !hal.device) flags("None") : !hal.fence
  %fence = hal.fence.create device(%arg0 : !hal.device) flags("None") : !hal.fence
  util.return %fence : !hal.fence
}

// -----

// CHECK-LABEL: @fence_join
util.func public @fence_join(%arg0: !hal.fence, %arg1: !hal.fence) -> !hal.fence {
  // CHECK: = hal.fence.join at([%arg0, %arg1]) -> !hal.fence
  %fence = hal.fence.join at([%arg0, %arg1]) -> !hal.fence
  util.return %fence : !hal.fence
}

// -----

// CHECK-LABEL: @fence_query
util.func public @fence_query(%arg0: !hal.fence) -> i32 {
  // CHECK: = hal.fence.query<%arg0 : !hal.fence> : i32
  %status = hal.fence.query<%arg0 : !hal.fence> : i32
  util.return %status : i32
}

// -----

// CHECK-LABEL: @fence_signal
util.func public @fence_signal(%arg0: !hal.fence) {
  // CHECK: hal.fence.signal<%arg0 : !hal.fence>
  hal.fence.signal<%arg0 : !hal.fence>
  util.return
}

// -----

// CHECK-LABEL: @fence_fail
util.func public @fence_fail(%arg0: !hal.fence, %arg1: i32) {
  // CHECK: hal.fence.fail<%arg0 : !hal.fence> status(%arg1)
  hal.fence.fail<%arg0 : !hal.fence> status(%arg1)
  util.return
}

// -----

// CHECK-LABEL: @fence_await
util.func public @fence_await(%arg0: !hal.fence, %arg1: !hal.fence, %arg2: i32) -> i32 {
  // CHECK: = hal.fence.await until([%arg0, %arg1]) timeout_millis(%arg2) : i32
  %status = hal.fence.await until([%arg0, %arg1]) timeout_millis(%arg2) : i32
  util.return %status : i32
}
