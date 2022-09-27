// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// Tests that a fence with no timepoints gets turned into a null value.
// This avoids the allocation and lets the null propagate through the rest of
// the program to simplify submissions.

// CHECK-LABEL: @fence_create_unused
func.func @fence_create_unused(%device: !hal.device) {
  // CHECK-NOT: hal.fence.create
  %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
  return
}

// -----

// Tests that a fence join with no operands folds into a util.null.

// CHECK-LABEL: @fence_join_empty
func.func @fence_join_empty() -> !hal.fence {
  // CHECK: %[[JOIN:.+]] = util.null : !hal.fence
  %join = hal.fence.join at([]) -> !hal.fence
  // CHECK: return %[[JOIN]]
  return %join : !hal.fence
}

// -----

// Tests that known null fences are dropped from joins.

// CHECK-LABEL: @fence_join_null
// CHECK-SAME: %[[ARG:.+]]: !hal.fence
func.func @fence_join_null(%arg: !hal.fence) -> !hal.fence {
  // CHECK-NOT: util.null
  %null = util.null : !hal.fence
  // CHECK: %[[JOIN:.+]] = hal.fence.join at([%[[ARG]]]) -> !hal.fence
  %join = hal.fence.join at([%arg, %null]) -> !hal.fence
  // CHECK: return %[[JOIN]]
  return %join : !hal.fence
}

// -----

// Tests deduplication of fences during a fence join.

// CHECK-LABEL: @fence_join_duplicate_fences
// CHECK-SAME: %[[FENCE0:.+]]: !hal.fence, %[[FENCE1:.+]]: !hal.fence
func.func @fence_join_duplicate_fences(%fence0: !hal.fence, %fence1: !hal.fence) -> !hal.fence {
  // CHECK: %[[JOIN:.+]] = hal.fence.join at([%[[FENCE0]], %[[FENCE1]]]) -> !hal.fence
  %join = hal.fence.join at([%fence0, %fence1, %fence0]) -> !hal.fence
  // CHECK: return %[[JOIN]]
  return %join : !hal.fence
}

// -----

// Tests that awaits with no fences are elided.

// CHECK-LABEL: @fence_await_none
func.func @fence_await_none() -> i32 {
  %timeout = arith.constant 123 : i32
  // CHECK: %[[STATUS:.+]] = arith.constant 0 : i32
  %status = hal.fence.await until([]) timeout_millis(%timeout) : i32
  // CHECK: return %[[STATUS]]
  return %status : i32
}

// -----

// Tests that known null fences are dropped from awaits.

// CHECK-LABEL: @fence_await_null
// CHECK-SAME: %[[ARG:.+]]: !hal.fence
func.func @fence_await_null(%arg: !hal.fence) -> i32 {
  %timeout = arith.constant 123 : i32
  // CHECK-NOT: util.null
  %null = util.null : !hal.fence
  // CHECK: %[[STATUS:.+]] = hal.fence.await until([%[[ARG]]])
  %status = hal.fence.await until([%arg, %null]) timeout_millis(%timeout) : i32
  // CHECK: return %[[STATUS]]
  return %status : i32
}

// -----

// Tests deduplication of fences during a fence await.

// CHECK-LABEL: @fence_await_duplicate_fences
// CHECK-SAME: %[[FENCE0:.+]]: !hal.fence, %[[FENCE1:.+]]: !hal.fence
func.func @fence_await_duplicate_fences(%fence0: !hal.fence, %fence1: !hal.fence) -> i32 {
  %timeout = arith.constant 123 : i32
  // CHECK: %[[STATUS:.+]] = hal.fence.await until([%[[FENCE0]], %[[FENCE1]]])
  %status = hal.fence.await until([%fence0, %fence1, %fence0]) timeout_millis(%timeout) : i32
  // CHECK: return %[[STATUS]]
  return %status : i32
}
