// RUN: iree-opt --split-input-file --iree-hal-outline-memoize-regions %s | FileCheck %s

// Tests that we don't memoize regions that we can't analyze.
// The memoize region will be outlined but called each use instead of memoized
// as global values.

// CHECK-LABEL: util.func private @__no_memoize_without_device_analysis_memoize_apply
//  CHECK-SAME: (%[[APPLY_DEVICE:.+]]: !hal.device, %[[APPLY_AFFINITY:.+]]: i64) -> (index, index)
//  CHECK-DAG: %[[C4:.+]] = arith.constant 4
//  CHECK-DAG: %[[C5:.+]] = arith.constant 5
// CHECK-NEXT: util.return %[[C4]], %[[C5]]

// CHECK-LABEL: util.func public @no_memoize_without_device_analysis
util.func public @no_memoize_without_device_analysis(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64)
    %device: !hal.device, %affinity: i64) -> (index, index) {
  // CHECK-NOT: hal.device.memoize
  // CHECK: %[[RESULTS:.+]]:2 = util.call @__no_memoize_without_device_analysis_memoize_apply(%[[DEVICE]], %[[AFFINITY]])
  %results:2 = hal.device.memoize<%device : !hal.device> affinity(%affinity) -> index, index {
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    hal.return %c4, %c5 : index, index
  }
  // CHECK: util.return %[[RESULTS]]#0, %[[RESULTS]]#1
  util.return %results#0, %results#1 : index, index
}

// -----

// Tests that we don't memoize regions that capture dynamic values.
// We'll still outline them but call them each use instead of memozing them in
// globals.

util.global private @device = #hal.device.target<"local"> : !hal.device

// CHECK-LABEL: util.func private @__no_memoize_with_dynamic_values_memoize_apply
//  CHECK-SAME: (%[[APPLY_DEVICE:.+]]: !hal.device, %[[APPLY_AFFINITY:.+]]: i64, %[[APPLY_ARG0:.+]]: index) -> (index, index)
//  CHECK-DAG: %[[C5:.+]] = arith.constant 5
// CHECK-NEXT: util.return %[[APPLY_ARG0]], %[[C5]]

// CHECK-LABEL: util.func public @no_memoize_with_dynamic_values
//  CHECK-SAME: (%[[ARG0:.+]]: index)
util.func public @no_memoize_with_dynamic_values(%arg0: index) -> (index, index) {
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  %device = util.global.load immutable @device : !hal.device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  %affinity = arith.constant -1 : i64
  // CHECK-NOT: hal.device.memoize
  // CHECK: %[[RESULTS:.+]]:2 = util.call @__no_memoize_with_dynamic_values_memoize_apply(%[[DEVICE]], %[[AFFINITY]], %[[ARG0]])
  %results:2 = hal.device.memoize<%device : !hal.device> affinity(%affinity) -> index, index {
    %c5 = arith.constant 5 : index
    hal.return %arg0, %c5 : index, index
  }
  // CHECK: util.return %[[RESULTS]]#0, %[[RESULTS]]#1
  util.return %results#0, %results#1 : index, index
}

// -----

// Tests memoizing a region that returns constant values.
// This is to test multiple results and non-HAL results.
// We expect the memoize apply function, globals for each device, and a lookup
// function to be created.

// CHECK: util.global private @device
util.global private @device = #hal.device.target<"local"> : !hal.device

// CHECK-LABEL: util.func private @__memoize_constants_memoize_apply
//  CHECK-SAME: (%[[APPLY_DEVICE:.+]]: !hal.device, %[[APPLY_AFFINITY:.+]]: i64) -> (index, index)
//  CHECK-DAG: %[[C4:.+]] = arith.constant 4
//  CHECK-DAG: %[[C5:.+]] = arith.constant 5
// CHECK-NEXT: util.return %[[C4]], %[[C5]]

// CHECK: util.global private @__memoize_constants_memoize_result_0_device : index
// CHECK: util.global private @__memoize_constants_memoize_result_1_device : index
// CHECK: util.initializer
//  CHECK-DAG: %[[INITIALIZER_DEVICE:.+]] = util.global.load immutable @device
//  CHECK-DAG: %[[INITIALIZER_AFFINITY:.+]] = arith.constant -1
// CHECK-NEXT: %[[APPLY_RESULTS:.+]]:2 = util.call @__memoize_constants_memoize_apply(%[[INITIALIZER_DEVICE]], %[[INITIALIZER_AFFINITY]])
// CHECK-NEXT: util.global.store %[[APPLY_RESULTS]]#0, @__memoize_constants_memoize_result_0_device
// CHECK-NEXT: util.global.store %[[APPLY_RESULTS]]#1, @__memoize_constants_memoize_result_1_device
// CHECK-NEXT: util.return

// CHECK-LABEL: util.func private @__memoize_constants_memoize_lookup
//  CHECK-SAME: (%[[LOOKUP_DEVICE:.+]]: !hal.device, %[[LOOKUP_AFFINITY:.+]]: i64) -> (index, index)
// CHECK-NEXT: %[[TRY_DEVICE:.+]] = util.global.load @device
// CHECK-NEXT: %[[IS_DEVICE:.+]] = util.cmp.eq %[[LOOKUP_DEVICE]], %[[TRY_DEVICE]]
//      CHECK: %[[IF_RESULTS:.+]]:2 = scf.if %[[IS_DEVICE]] -> (index, index)
//  CHECK-DAG:   %[[DEVICE_RESULT_0:.+]] = util.global.load @__memoize_constants_memoize_result_0_device
//  CHECK-DAG:   %[[DEVICE_RESULT_1:.+]] = util.global.load @__memoize_constants_memoize_result_1_device
// CHECK-NEXT:   scf.yield %[[DEVICE_RESULT_0]], %[[DEVICE_RESULT_1]]
// CHECK-NEXT: } else {
//  CHECK-DAG:   %[[FALLBACK_RESULT_0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[FALLBACK_RESULT_1:.+]] = arith.constant 0
// CHECK-NEXT:   scf.yield %[[FALLBACK_RESULT_0]], %[[FALLBACK_RESULT_1]]
// CHECK: util.return %[[IF_RESULTS]]#0, %[[IF_RESULTS]]#1

// CHECK-LABEL: util.func public @memoize_constants
util.func public @memoize_constants() -> (index, index) {
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  %device = util.global.load immutable @device : !hal.device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  %affinity = arith.constant -1 : i64
  // CHECK-NOT: hal.device.memoize
  // CHECK: %[[LOOKUP_RESULTS:.+]]:2 = util.call @__memoize_constants_memoize_lookup(%[[DEVICE]], %[[AFFINITY]])
  %result:2 = hal.device.memoize<%device : !hal.device> affinity(%affinity) -> index, index {
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    hal.return %c4, %c5 : index, index
  }
  // CHECK: util.return %[[LOOKUP_RESULTS]]#0, %[[LOOKUP_RESULTS]]#1
  util.return %result#0, %result#1 : index, index
}

// -----

// Tests that each device that may use a memoized value gets its own copy.

util.global private @device_a = #hal.device.target<"local", {ordinal = 0 : index}> : !hal.device
util.global private @device_b = #hal.device.target<"local", {ordinal = 1 : index}> : !hal.device

// CHECK: util.func private @__memoize_multiple_devices_memoize_apply

// CHECK-LABEL: util.global private @__memoize_multiple_devices_memoize_result_0_device_a : index
// CHECK-NEXT: util.initializer
//  CHECK-DAG: %[[INITIALIZER_DEVICE_A:.+]] = util.global.load immutable @device_a
//  CHECK-DAG: %[[INITIALIZER_AFFINITY_A:.+]] = arith.constant -1
//  CHECK-DAG: %[[INITIALIZER_RESULT_A:.+]] = util.call @__memoize_multiple_devices_memoize_apply(%[[INITIALIZER_DEVICE_A]], %[[INITIALIZER_AFFINITY_A]])
//  CHECK-DAG: util.global.store %[[INITIALIZER_RESULT_A]], @__memoize_multiple_devices_memoize_result_0_device_a

// CHECK-LABEL: util.global private @__memoize_multiple_devices_memoize_result_0_device_b : index
// CHECK-NEXT: util.initializer
//  CHECK-DAG: %[[INITIALIZER_DEVICE_B:.+]] = util.global.load immutable @device_b
//  CHECK-DAG: %[[INITIALIZER_AFFINITY_B:.+]] = arith.constant -1
//  CHECK-DAG: %[[INITIALIZER_RESULT_B:.+]] = util.call @__memoize_multiple_devices_memoize_apply(%[[INITIALIZER_DEVICE_B]], %[[INITIALIZER_AFFINITY_B]])
//  CHECK-DAG: util.global.store %[[INITIALIZER_RESULT_B]], @__memoize_multiple_devices_memoize_result_0_device_b

// CHECK-LABEL: util.func private @__memoize_multiple_devices_memoize_lookup
//  CHECK-SAME: (%[[LOOKUP_DEVICE:.+]]: !hal.device, %[[LOOKUP_AFFINITY:.+]]: i64) -> index
// CHECK-NEXT: %[[TRY_DEVICE_A:.+]] = util.global.load @device_a
// CHECK-NEXT: %[[IS_DEVICE_A:.+]] = util.cmp.eq %[[LOOKUP_DEVICE]], %[[TRY_DEVICE_A]]
//      CHECK: %[[IF_RESULT_A:.+]] = scf.if %[[IS_DEVICE_A]] -> (index)
// CHECK-NEXT:   %[[LOOKUP_RESULT_A:.+]] = util.global.load @__memoize_multiple_devices_memoize_result_0_device_a
// CHECK-NEXT:   scf.yield %[[LOOKUP_RESULT_A]]
// CHECK-NEXT: } else {
// CHECK-NEXT:   %[[TRY_DEVICE_B:.+]] = util.global.load @device_b
// CHECK-NEXT:   %[[IS_DEVICE_B:.+]] = util.cmp.eq %[[LOOKUP_DEVICE]], %[[TRY_DEVICE_B]]
// CHECK-NEXT:   %[[IF_RESULT_B:.+]] = scf.if %[[IS_DEVICE_B]] -> (index)
// CHECK-NEXT:     %[[LOOKUP_RESULT_B:.+]] = util.global.load @__memoize_multiple_devices_memoize_result_0_device_b
// CHECK-NEXT:     scf.yield %[[LOOKUP_RESULT_B]]
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %[[FALLBACK_RESULT:.+]] = arith.constant 0
// CHECK-NEXT:     scf.yield %[[FALLBACK_RESULT]]
//      CHECK:   scf.yield %[[IF_RESULT_B]]
//      CHECK: util.return %[[IF_RESULT_A]]

// CHECK-LABEL: util.func public @memoize_multiple_devices
// CHECK-SAME: (%[[SELECTOR:.+]]: i1)
util.func public @memoize_multiple_devices(%selector: i1) -> index {
  // CHECK-DAG: %[[DEVICE_A:.+]] = util.global.load immutable @device_a
  %device_a = util.global.load immutable @device_a : !hal.device
  // CHECK-DAG: %[[AFFINITY_A:.+]] = arith.constant 1
  %affinity_a = arith.constant 1 : i64
  // CHECK-DAG: %[[DEVICE_B:.+]] = util.global.load immutable @device_b
  %device_b = util.global.load immutable @device_b : !hal.device
  // CHECK-DAG: %[[AFFINITY_B:.+]] = arith.constant 3
  %affinity_b = arith.constant 3 : i64
  // CHECK-DAG: %[[DEVICE:.+]] = arith.select %[[SELECTOR]], %[[DEVICE_A]], %[[DEVICE_B]]
  %device = arith.select %selector, %device_a, %device_b : !hal.device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.select %[[SELECTOR]], %[[AFFINITY_A]], %[[AFFINITY_B]]
  %affinity = arith.select %selector, %affinity_a, %affinity_b : i64
  // CHECK-NOT: hal.device.memoize
  // CHECK: %[[RESULT:.+]] = util.call @__memoize_multiple_devices_memoize_lookup(%[[DEVICE]], %[[AFFINITY]])
  %result = hal.device.memoize<%device : !hal.device> affinity(%affinity) -> index {
    %c4 = arith.constant 4 : index
    hal.return %c4 : index
  }
  // CHECK: util.return %[[RESULT]]
  util.return %result : index
}

// -----

// Tests memoization of a command buffer that uses global resources.

util.global private @device = #hal.device.target<"local"> : !hal.device
util.global private @executable : !hal.executable
util.global private @buffer : !hal.buffer

// CHECK-LABEL: util.func private @__memoize_command_buffer_memoize_apply
//  CHECK-SAME: (%[[APPLY_DEVICE:.+]]: !hal.device, %[[APPLY_AFFINITY:.+]]: i64, %[[APPLY_EXECUTABLE:.+]]: !hal.executable, %[[APPLY_BUFFER:.+]]: !hal.buffer) -> !hal.command_buffer
//       CHECK: %[[CMD:.+]] = hal.command_buffer.create device(%[[APPLY_DEVICE]] : !hal.device)
//  CHECK-SAME:   affinity(%[[APPLY_AFFINITY]])
//       CHECK: hal.command_buffer.dispatch.indirect<%[[CMD]] : !hal.command_buffer>
//  CHECK-SAME:   target(%[[APPLY_EXECUTABLE]] : !hal.executable)
//  CHECK-SAME:   workgroups(%[[APPLY_BUFFER]] : !hal.buffer)
//       CHECK: hal.command_buffer.execution_barrier
//       CHECK: hal.command_buffer.finalize
//       CHECK: util.return %[[CMD]]

// CHECK-LABEL: util.global private @__memoize_command_buffer_memoize_result_0_device : !hal.command_buffer
//  CHECK-NEXT: util.initializer
//   CHECK-DAG: %[[INITIALIZER_DEVICE:.+]] = util.global.load immutable @device
//   CHECK-DAG: %[[INITIALIZER_AFFINITY:.+]] = arith.constant 2
//   CHECK-DAG: %[[INITIALIZER_EXECUTABLE:.+]] = util.global.load immutable @executable
//   CHECK-DAG: %[[INITIALIZER_BUFFER:.+]] = util.global.load immutable @buffer
//  CHECK-NEXT: %[[APPLY_RESULT:.+]] = util.call @__memoize_command_buffer_memoize_apply(%[[INITIALIZER_DEVICE]], %[[INITIALIZER_AFFINITY]], %[[INITIALIZER_EXECUTABLE]], %[[INITIALIZER_BUFFER]])
//  CHECK-NEXT: util.global.store %[[APPLY_RESULT]], @__memoize_command_buffer_memoize_result_0_device
//  CHECK-NEXT: util.return

// CHECK-LABEL: util.func private @__memoize_command_buffer_memoize_lookup

// CHECK-LABEL: util.func public @memoize_command_buffer
util.func public @memoize_command_buffer() -> !hal.command_buffer {
  %device = util.global.load immutable @device : !hal.device
  %affinity = arith.constant 2 : i64
  %executable = util.global.load immutable @executable : !hal.executable
  %buffer = util.global.load immutable @buffer : !hal.buffer
  // CHECK-NOT: hal.device.memoize
  // CHECK: %[[CMD:.+]] = util.call @__memoize_command_buffer_memoize_lookup
  %result = hal.device.memoize<%device : !hal.device> affinity(%affinity) -> !hal.command_buffer {
    %cmd = hal.command_buffer.create device(%device : !hal.device) mode(None) categories("Transfer|Dispatch") affinity(%affinity) : !hal.command_buffer
    %dispatch_ordinal = arith.constant 123 : index
    %offset = arith.constant 100 : index
    hal.command_buffer.dispatch.indirect<%cmd : !hal.command_buffer>
        target(%executable : !hal.executable)[%dispatch_ordinal]
        workgroups(%buffer : !hal.buffer)[%offset]
        flags(None)
    hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer>
        source(CommandIssue) target(CommandProcess) flags(None)
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    hal.return %cmd : !hal.command_buffer
  }
  // CHECK: util.return %[[CMD]]
  util.return %result : !hal.command_buffer
}
