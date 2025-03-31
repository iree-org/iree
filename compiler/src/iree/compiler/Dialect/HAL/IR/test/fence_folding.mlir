// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// Tests that a fence with no timepoints gets turned into a null value.
// This avoids the allocation and lets the null propagate through the rest of
// the program to simplify submissions.

// CHECK-LABEL: @fence_create_unused
util.func public @fence_create_unused(%device: !hal.device) {
  // CHECK-NOT: hal.fence.create
  %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
  util.return
}

// -----

// Tests that a fence join with 1 operand folds into that operand.

// CHECK-LABEL: @fence_join_one
// CHECK-SAME: %[[ARG:.+]]: !hal.fence
util.func public @fence_join_one(%arg: !hal.fence) -> !hal.fence {
  %join = hal.fence.join at([%arg]) flags("None") -> !hal.fence
  // CHECK: util.return %[[ARG]]
  util.return %join : !hal.fence
}

// -----

// Tests that a fence join with no operands folds into a util.null.

// CHECK-LABEL: @fence_join_empty
util.func public @fence_join_empty() -> !hal.fence {
  // CHECK: %[[JOIN:.+]] = util.null : !hal.fence
  %join = hal.fence.join at([]) flags("None") -> !hal.fence
  // CHECK: util.return %[[JOIN]]
  util.return %join : !hal.fence
}

// -----

// Tests that known null fences are dropped from joins.

// CHECK-LABEL: @fence_join_null
// CHECK-SAME: (%[[ARG0:.+]]: !hal.fence, %[[ARG1:.+]]: !hal.fence)
util.func public @fence_join_null(%arg0: !hal.fence, %arg1: !hal.fence) -> !hal.fence {
  // CHECK-NOT: util.null
  %null = util.null : !hal.fence
  // CHECK: %[[JOIN:.+]] = hal.fence.join at([%[[ARG0]], %[[ARG1]]]) flags("None") -> !hal.fence
  %join = hal.fence.join at([%arg0, %null, %arg1]) flags("None") -> !hal.fence
  // CHECK: util.return %[[JOIN]]
  util.return %join : !hal.fence
}

// -----

// Tests deduplication of fences during a fence join.

// CHECK-LABEL: @fence_join_duplicate_fences
// CHECK-SAME: %[[FENCE0:.+]]: !hal.fence, %[[FENCE1:.+]]: !hal.fence
util.func public @fence_join_duplicate_fences(%fence0: !hal.fence, %fence1: !hal.fence) -> !hal.fence {
  // CHECK: %[[JOIN:.+]] = hal.fence.join at([%[[FENCE0]], %[[FENCE1]]]) flags("None") -> !hal.fence
  %join = hal.fence.join at([%fence0, %fence1, %fence0]) flags("None") -> !hal.fence
  // CHECK: util.return %[[JOIN]]
  util.return %join : !hal.fence
}

// -----

// Elides fences that are immediately signaled on the host.
// This requires that there are no ops using the fence value between the time it
// is created and the time it is signaled.

// CHECK-LABEL: @fence_elide_signaled
util.func public @fence_elide_signaled(%device: !hal.device) -> !hal.fence {
  // CHECK: %[[NULL_FENCE:.+]] = util.null : !hal.fence
  // CHECK-NOT: hal.fence.create
  %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
  // Ok to have other things inbetween so long as they don't touch the fence.
  // CHECK: util.call @external_nop_call
  util.call @external_nop_call() : () -> ()
  // CHECK-NOT: hal.fence.signal
  hal.fence.signal<%fence : !hal.fence>
  // CHECK: util.return %[[NULL_FENCE]]
  util.return %fence : !hal.fence
}
util.func private @external_nop_call()

// -----

// Ensures that immediate fence signals aren't elided if the fence may be waited
// on between when it is created and when it is signaled.

// CHECK-LABEL: @fence_cannot_elide_signaled
util.func public @fence_cannot_elide_signaled(%device: !hal.device) -> !hal.fence {
  // CHECK: hal.fence.create
  %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
  // Block the elision as the external call may wait on the fence.
  // CHECK: util.call @external_wait_call
  util.call @external_wait_call(%fence) : (!hal.fence) -> ()
  // CHECK: hal.fence.signal
  hal.fence.signal<%fence : !hal.fence>
  // CHECK: util.return
  util.return %fence : !hal.fence
}
util.func private @external_wait_call(!hal.fence)

// -----

// Tests that awaits with no fences are elided.

// CHECK-LABEL: @fence_await_none
util.func public @fence_await_none() -> i32 {
  %timeout = arith.constant 123 : i32
  // CHECK: %[[STATUS:.+]] = arith.constant 0 : i32
  %status = hal.fence.await until([]) timeout_millis(%timeout) flags("None") : i32
  // CHECK: util.return %[[STATUS]]
  util.return %status : i32
}

// -----

// Tests that known null fences are dropped from awaits.

// CHECK-LABEL: @fence_await_null
// CHECK-SAME: %[[ARG:.+]]: !hal.fence
util.func public @fence_await_null(%arg: !hal.fence) -> i32 {
  %timeout = arith.constant 123 : i32
  // CHECK-NOT: util.null
  %null = util.null : !hal.fence
  // CHECK: %[[STATUS:.+]] = hal.fence.await until([%[[ARG]]])
  %status = hal.fence.await until([%arg, %null]) timeout_millis(%timeout) flags("None") : i32
  // CHECK: util.return %[[STATUS]]
  util.return %status : i32
}

// -----

// Tests deduplication of fences during a fence await.

// CHECK-LABEL: @fence_await_duplicate_fences
// CHECK-SAME: %[[FENCE0:.+]]: !hal.fence, %[[FENCE1:.+]]: !hal.fence
util.func public @fence_await_duplicate_fences(%fence0: !hal.fence, %fence1: !hal.fence) -> i32 {
  %timeout = arith.constant 123 : i32
  // CHECK: %[[STATUS:.+]] = hal.fence.await until([%[[FENCE0]], %[[FENCE1]]])
  %status = hal.fence.await until([%fence0, %fence1, %fence0]) timeout_millis(%timeout) flags("None") : i32
  // CHECK: util.return %[[STATUS]]
  util.return %status : i32
}
