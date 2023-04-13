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

// Tests that a fence join with 1 operand folds into that operand.

// CHECK-LABEL: @fence_join_one
// CHECK-SAME: %[[ARG:.+]]: !hal.fence
func.func @fence_join_one(%arg: !hal.fence) -> !hal.fence {
  %join = hal.fence.join at([%arg]) -> !hal.fence
  // CHECK: return %[[ARG]]
  return %join : !hal.fence
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
// CHECK-SAME: (%[[ARG0:.+]]: !hal.fence, %[[ARG1:.+]]: !hal.fence)
func.func @fence_join_null(%arg0: !hal.fence, %arg1: !hal.fence) -> !hal.fence {
  // CHECK-NOT: util.null
  %null = util.null : !hal.fence
  // CHECK: %[[JOIN:.+]] = hal.fence.join at([%[[ARG0]], %[[ARG1]]]) -> !hal.fence
  %join = hal.fence.join at([%arg0, %null, %arg1]) -> !hal.fence
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

// Elides fences that are immediately signaled on the host.
// This requires that there are no ops using the fence value between the time it
// is created and the time it is signaled.

// CHECK-LABEL: @fence_elide_signaled
func.func @fence_elide_signaled(%device: !hal.device) -> !hal.fence {
  // CHECK-NOT: hal.fence.create
  %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
  // Ok to have other things inbetween so long as they don't touch the fence.
  // CHECK: call @external_nop_call
  call @external_nop_call() : () -> ()
  // CHECK-NOT: hal.fence.signal
  hal.fence.signal<%fence : !hal.fence>
  // CHECK: %[[FENCE:.+]] = util.null : !hal.fence
  // CHECK: return %[[FENCE]]
  return %fence : !hal.fence
}
func.func private @external_nop_call()

// -----

// Ensures that immediate fence signals aren't elided if the fence may be waited
// on between when it is created and when it is signaled.

// CHECK-LABEL: @fence_cannot_elide_signaled
func.func @fence_cannot_elide_signaled(%device: !hal.device) -> !hal.fence {
  // CHECK: hal.fence.create
  %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
  // Block the elision as the external call may wait on the fence.
  // CHECK: call @external_wait_call
  call @external_wait_call(%fence) : (!hal.fence) -> ()
  // CHECK: hal.fence.signal
  hal.fence.signal<%fence : !hal.fence>
  // CHECK: return
  return %fence : !hal.fence
}
func.func private @external_wait_call(!hal.fence)

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
