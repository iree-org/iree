// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// Tests that a fence with no timepoints gets turned into a null value.
// This avoids the allocation and lets the null propagate through the rest of
// the program to simplify submissions.

// CHECK-LABEL: @fence_create_empty
func.func @fence_create_empty() -> !hal.fence {
  // CHECK: %[[FENCE:.+]] = util.null : !hal.fence
  %fence = hal.fence.create -> !hal.fence
  // CHECK: return %[[FENCE]]
  return %fence : !hal.fence
}

// -----

// Tests that a fence with multiple timepoints sharing the same semaphore are
// deduplicated to the max of the timepoints.

// CHECK-LABEL: @fence_create_duplicate_semaphores
// CHECK-SAME: %[[SEMAPHORE0:.+]]: !hal.semaphore, %[[TIME0:.+]]: i64, %[[SEMAPHORE1:.+]]: !hal.semaphore, %[[TIME1:.+]]: i64, %[[TIME2:.+]]: i64
func.func @fence_create_duplicate_semaphores(
    %semaphore0: !hal.semaphore, %time0: i64,
    %semaphore1: !hal.semaphore, %time1: i64, %time2: i64) -> !hal.fence {
  // CHECK: %[[TIMEMAX:.+]] = arith.maxui %[[TIME1]], %[[TIME2]] : i64
  // CHECK: %[[FENCE:.+]] = hal.fence.create
  // CHECK-SAME: at<%[[SEMAPHORE0]] : !hal.semaphore>(%[[TIME0]])
  // CHECK-SAME: at<%[[SEMAPHORE1]] : !hal.semaphore>(%[[TIMEMAX]])
  %fence = hal.fence.create
      at<%semaphore0 : !hal.semaphore>(%time0)
      at<%semaphore1 : !hal.semaphore>(%time1)
      at<%semaphore1 : !hal.semaphore>(%time2)
      -> !hal.fence
  // CHECK: return %[[FENCE]]
  return %fence : !hal.fence
}

// -----

// Tests that timepoints with the same values are deduplicated.
// This would be handled by util.range.max canonicalizations as above but this
// avoids emitting additional IR and is effectively free.

// CHECK-LABEL: @fence_create_duplicate_values
// CHECK-SAME: %[[SEMAPHORE0:.+]]: !hal.semaphore, %[[TIME0:.+]]: i64, %[[SEMAPHORE1:.+]]: !hal.semaphore, %[[TIME1:.+]]: i64
func.func @fence_create_duplicate_values(
    %semaphore0: !hal.semaphore, %time0: i64,
    %semaphore1: !hal.semaphore, %time1: i64) -> !hal.fence {
  // CHECK: %[[FENCE:.+]] = hal.fence.create
  %fence = hal.fence.create
      // CHECK-SAME: at<%[[SEMAPHORE0]] : !hal.semaphore>(%[[TIME0]])
      at<%semaphore0 : !hal.semaphore>(%time0)
      at<%semaphore0 : !hal.semaphore>(%time0)
      at<%semaphore0 : !hal.semaphore>(%time0)
      // CHECK-SAME: at<%[[SEMAPHORE1]] : !hal.semaphore>(%[[TIME1]])
      at<%semaphore1 : !hal.semaphore>(%time1)
      at<%semaphore1 : !hal.semaphore>(%time1)
      -> !hal.fence
  // CHECK: return %[[FENCE]]
  return %fence : !hal.fence
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
