// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.initializer(iree-stream-automatic-reference-counting),util.func(iree-stream-automatic-reference-counting))' %s | FileCheck %s

// Producers return references initialized to 1 and we only need to release them
// at last use.

// CHECK-LABEL: @execute_producer
util.func private @execute_producer() {
  %c123_i32 = arith.constant 123 : i32
  %size = arith.constant 1024 : index
  %splat, %splat_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<transient>{%size} {
    %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.return
}

// -----

// Consumers don't technically need to change reference counts but we
// conservatively insert retains to balance releases so that we can track the
// timeline where we could insert deallocations. We rely on whole-program
// analysis to remove unneeded retains/releases.

// CHECK-LABEL: @execute_consumer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[TIMEPOINT:.+]]: !stream.timepoint)
util.func private @execute_consumer(%resource: !stream.resource<transient>, %size: index, %timepoint: !stream.timepoint) {
  %transfer, %transfer_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) await(%timepoint) => with(%resource as %resource_capture: !stream.resource<transient>{%size}) -> !stream.resource<external>{%size} {
    %0 = stream.async.transfer %resource_capture : !stream.resource<transient>{%size} -> !stream.resource<external>{%size}
    stream.yield %0 : !stream.resource<external>{%size}
  } => !stream.timepoint
  util.return
}

// -----

// CHECK-LABEL: @execute_consumers
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[TIMEPOINT_A:.+]]: !stream.timepoint, %[[TIMEPOINT_B:.+]]: !stream.timepoint)
util.func private @execute_consumers(%resource: !stream.resource<transient>, %size: index, %timepoint_a: !stream.timepoint, %timepoint_b: !stream.timepoint) {
  %transfer_a, %transfer_a_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) await(%timepoint_a) => with(%resource as %resource_capture: !stream.resource<transient>{%size}) -> !stream.resource<external>{%size} {
    %0 = stream.async.transfer %resource_capture : !stream.resource<transient>{%size} -> !stream.resource<external>{%size}
    stream.yield %0 : !stream.resource<external>{%size}
  } => !stream.timepoint
  %transfer_b, %transfer_b_timepoint = stream.async.execute on(#hal.device.promise<@dev_b>) await(%timepoint_b) => with(%resource as %resource_capture: !stream.resource<transient>{%size}) -> !stream.resource<external>{%size} {
    %0 = stream.async.transfer %resource_capture : !stream.resource<transient>{%size} -> !stream.resource<external>{%size}
    stream.yield %0 : !stream.resource<external>{%size}
  } => !stream.timepoint
  util.return
}

// -----

// Tied operands are handled special as they are creating a new SSA value for an
// existing resource. We do some optimization inline to avoid extra IR.

// CHECK-LABEL: @execute_tied
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[TIMEPOINT:.+]]: !stream.timepoint)
util.func private @execute_tied(%resource: !stream.resource<transient>, %size: index, %timepoint: !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  %fill, %fill_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with(%resource as %resource_capture: !stream.resource<transient>{%size}) -> %resource{%size} {
    %0 = stream.async.fill %c123_i32, %resource_capture[%c0 to %size for %size] : i32 -> %resource_capture as !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.return
}

// -----

// Simple producer->consumer flows should balance retains and releases so that
// resources can be deallocated on their final use.

// CHECK-LABEL: @execute_producer_to_consumer
util.func private @execute_producer_to_consumer() {
  %size = arith.constant 1024 : index
  %c123_i32 = arith.constant 123 : i32
  %splat, %splat_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<transient>{%size} {
    %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  %transfer, %transfer_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) await(%splat_timepoint) => with(%splat as %splat_capture: !stream.resource<transient>{%size}) -> !stream.resource<external>{%size} {
    %0 = stream.async.transfer %splat_capture : !stream.resource<transient>{%size} -> !stream.resource<external>{%size}
    stream.yield %0 : !stream.resource<external>{%size}
  } => !stream.timepoint
  util.return
}

// -----

// Tests that imports retain the imported resource by default.

// CHECK-LABEL: @import_retain
util.func private @import_retain(%buffer_view: !hal.buffer_view, %fence: !hal.fence) {
  // CHECK: %[[SIZE:.+]] = arith.constant 1024
  %size = arith.constant 1024 : index
  // CHECK: %[[IMPORTED_RESOURCE:.+]] = stream.tensor.import
  // CHECK-NEXT: %[[RETAINED_RESOURCE:.+]] = stream.async.retain %[[IMPORTED_RESOURCE]] : !stream.resource<external>{%[[SIZE]]}
  %imported_resource = stream.tensor.import on(#hal.device.promise<@dev_a>) %buffer_view : !hal.buffer_view -> tensor<1xi32> in !stream.resource<external>{%size}
  // CHECK: %[[IMPORTED_TIMEPOINT:.+]] = stream.timepoint.import
  %imported_timepoint = stream.timepoint.import on(#hal.device.promise<@dev_a>) %fence : (!hal.fence) => !stream.timepoint
  // CHECK: stream.timepoint.await %[[IMPORTED_TIMEPOINT]] => %[[RETAINED_RESOURCE]]
  %ready_resource = stream.timepoint.await %imported_timepoint => %imported_resource : !stream.resource<external>{%size}
  util.return
}

// -----

// Tests that imports marked as consuming the resource do not retain it.
// This allows for ownership transfer.

// CHECK-LABEL: @import_transfer
util.func private @import_transfer(%buffer_view: !hal.buffer_view, %fence: !hal.fence) {
  // CHECK: %[[SIZE:.+]] = arith.constant 1024
  %size = arith.constant 1024 : index
  // CHECK-NOT: stream.async.retain
  %imported_resource = stream.tensor.import on(#hal.device.promise<@dev_a>) consume %buffer_view : !hal.buffer_view -> tensor<1xi32> in !stream.resource<external>{%size}
  %imported_timepoint = stream.timepoint.import on(#hal.device.promise<@dev_a>) %fence : (!hal.fence) => !stream.timepoint
  %ready_resource = stream.timepoint.await %imported_timepoint => %imported_resource : !stream.resource<external>{%size}
  util.return
}

// -----

// Tests that exports retain the resource for the caller.

// CHECK-LABEL: @export_retain
util.func private @export_retain(%fence: !hal.fence) {
  %size = arith.constant 1024 : index
  %c123_i32 = arith.constant 123 : i32
  %result, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%size} {
    %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%size}
    stream.yield %0 : !stream.resource<external>{%size}
  } => !stream.timepoint
  stream.timepoint.chain_external on(#hal.device.promise<@dev_a>) %result_timepoint => (%fence : !hal.fence)
  %buffer_view = stream.tensor.export on(#hal.device.promise<@dev_a>) %result : tensor<1xi32> in !stream.resource<external>{%size} -> !hal.buffer_view
  util.return
}

// -----

util.global private @resource : !stream.resource<constant>
util.global private @size : index
util.func private @global_load() {
  %resource = util.global.load immutable @resource : !stream.resource<constant>
  %size = util.global.load immutable @size : index
  %transfer, %transfer_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with(%resource as %resource_capture: !stream.resource<constant>{%size}) -> !stream.resource<external>{%size} {
    %0 = stream.async.transfer %resource_capture : !stream.resource<constant>{%size} -> !stream.resource<external>{%size}
    stream.yield %0 : !stream.resource<external>{%size}
  } => !stream.timepoint
  util.return
}

// -----

util.global private mutable @resource_timepoint = #stream.timepoint<immediate> : !stream.timepoint
util.global private mutable @resource : !stream.resource<variable>
util.global private mutable @size : index
util.func private @global_replace() {
  %resource_timepoint = util.global.load @resource_timepoint : !stream.timepoint
  %resource = util.global.load @resource : !stream.resource<variable>
  %size = util.global.load @size : index
  %transfer, %transfer_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) await(%resource_timepoint) => with(%resource as %resource_capture: !stream.resource<variable>{%size}) -> !stream.resource<variable>{%size} {
    %0 = stream.async.transfer %resource_capture : !stream.resource<variable>{%size} -> !stream.resource<variable>{%size}
    stream.yield %0 : !stream.resource<variable>{%size}
  } => !stream.timepoint
  util.global.store %transfer, @resource : !stream.resource<variable>
  util.global.store %size, @size : index
  util.global.store %transfer_timepoint, @resource_timepoint : !stream.timepoint
  util.return
}

// -----

util.global private mutable @resource_timepoint = #stream.timepoint<immediate> : !stream.timepoint
util.global private @resource : !stream.resource<variable>
util.global private @size : index
util.func private @global_update_inplace() {
  %resource_timepoint = util.global.load @resource_timepoint : !stream.timepoint
  %resource = util.global.load immutable @resource : !stream.resource<variable>
  %size = util.global.load immutable @size : index
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  %fill, %fill_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with(%resource as %resource_capture: !stream.resource<variable>{%size}) -> %resource{%size} {
    %0 = stream.async.fill %c123_i32, %resource_capture[%c0 to %size for %size] : i32 -> %resource_capture as !stream.resource<variable>{%size}
    stream.yield %0 : !stream.resource<variable>{%size}
  } => !stream.timepoint
  util.global.store %fill_timepoint, @resource_timepoint : !stream.timepoint
  util.return
}

// -----

// CHECK-LABEL: @optimization_barrier
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<transient>)
util.func private @optimization_barrier(%resource: !stream.resource<transient>) {
  // CHECK: %[[BARRIER:.+]] = util.optimization_barrier %[[RESOURCE]]
  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[BARRIER]]
  // CHECK: %[[RETAINED_BARRIER:.+]] = stream.async.retain %[[BARRIER]] : !stream.resource<transient>{%[[SIZE]]}
  %barrier = util.optimization_barrier %resource : !stream.resource<transient>
  util.return
}

// -----

// CHECK-LABEL: @select
// CHECK-SAME: (%[[COND:.+]]: i1, %[[A:.+]]: !stream.resource<transient>, %[[B:.+]]: !stream.resource<transient>)
util.func private @select(%cond: i1, %a: !stream.resource<transient>, %b: !stream.resource<transient>) {
  // CHECK: %[[SELECTED:.+]] = arith.select %[[COND]], %[[A]], %[[B]]
  // CHECK: %[[RETAINED_SELECTED:.+]] = stream.async.retain %[[SELECTED]]
  %selected = arith.select %cond, %a, %b : !stream.resource<transient>
  util.return
}

// -----

// util.func private @call()

// // -----

// util.func private @return_duplicates()

// // -----

// util.func private @br()

// // -----

// util.func private @br_duplicates()

// // -----

// util.func private @cond_br_convergent()

// // -----

// util.func private @cond_br_divergent()

// // -----

// util.func private @cfg_loop()

// // -----

// util.func private @scf_if()

// // -----

// util.func private @scf_for()

// // -----

// util.func private @scf_while()



// util.func public @optimization_barrier_consumer() -> (!stream.resource<external>, index) {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %2 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   %1 = util.optimization_barrier %0 : !stream.resource<external>
//   util.return %1, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @optimization_barrier_producer() -> (!stream.resource<external>, index) {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %2 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   %1 = util.optimization_barrier %0 : !stream.resource<external>
//   util.return %1, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @constant_op() -> (!stream.resource<external>, index, !stream.resource<external>, index) {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results:2, %result_timepoint = stream.async.execute with() -> (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     %2:2 = stream.async.concurrent with(%1 as %arg0: !stream.resource<transient>{%c4}) -> (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) {
//       %3 = stream.async.transfer %arg0 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_a>) !stream.resource<external>{%c4}
//       %4 = stream.async.transfer %arg0 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_b>) !stream.resource<external>{%c4}
//       stream.yield %3, %4 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//     }
//     stream.yield %2#0, %2#1 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0:2 = stream.timepoint.await %result_timepoint => %results#0, %results#1 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//   util.return %0#0, %c4, %0#1, %c4 : !stream.resource<external>, index, !stream.resource<external>, index
// }

// // -----

// util.func public @splat_op() -> (!stream.resource<external>, index) {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %1 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   util.return %0, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @imported_hal_tensor(%arg0: !hal.buffer_view, %arg1: !hal.fence) -> (!stream.resource<external>, index) {
//   %c4 = arith.constant 4 : index
//   %c1 = arith.constant 1 : index
//   %element_type_i32 = hal.element_type<i32> : i32
//   %dense_row_major = hal.encoding_type<dense_row_major> : i32
//   hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input") shape([%c1]) type(%element_type_i32) encoding(%dense_row_major)
//   %0 = stream.tensor.import on(#hal.device.promise<@dev_a>) %arg0 : !hal.buffer_view -> tensor<1xi32> in !stream.resource<external>{%c4}
//   %1 = stream.timepoint.import on(#hal.device.promise<@dev_a>) %arg1 : (!hal.fence) => !stream.timepoint
//   %2 = stream.timepoint.await %1 => %0 : !stream.resource<external>{%c4}
//   util.return %2, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @imported_stream_tensor(%arg0: !hal.buffer_view) -> !stream.resource<external> {
//   %c4 = arith.constant 4 : index
//   %0 = stream.tensor.import on(#hal.device.promise<@dev_a>) %arg0 : !hal.buffer_view -> tensor<1xi32> in !stream.resource<external>{%c4}
//   util.return %0 : !stream.resource<external>
// }

// // -----

// util.func public @exported_hal_constant(%arg0: !hal.fence) -> !hal.buffer_view {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %1 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   stream.timepoint.chain_external on(#hal.device.promise<@dev_a>) %result_timepoint => (%arg0 : !hal.fence)
//   %0 = stream.tensor.export on(#hal.device.promise<@dev_a>) %results : tensor<1xi32> in !stream.resource<external>{%c4} -> !hal.buffer_view
//   util.return %0 : !hal.buffer_view
// }

// // -----

// util.func public @exported_stream_constant() -> !hal.buffer_view {
//   %c0_i8 = arith.constant 0 : i8
//   %c1280 = arith.constant 1280 : index
//   %results, %result_timepoint = stream.async.execute with() -> !stream.resource<external>{%c1280} {
//     %2 = stream.async.splat %c0_i8 : i8 -> !stream.resource<external>{%c1280}
//     stream.yield %2 : !stream.resource<external>{%c1280}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c1280}
//   %1 = stream.tensor.export on(#hal.device.promise<@dev_a>) %0 : tensor<1x5x64xi32> in !stream.resource<external>{%c1280} -> !hal.buffer_view
//   util.return %1 : !hal.buffer_view
// }

// // -----

// util.func public @exported_producer(%arg0: !hal.fence) -> !hal.buffer_view {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %1 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   stream.timepoint.chain_external on(#hal.device.promise<@dev_b>) %result_timepoint => (%arg0 : !hal.fence)
//   %0 = stream.tensor.export on(#hal.device.promise<@dev_b>) %results : tensor<1xi32> in !stream.resource<external>{%c4} -> !hal.buffer_view
//   util.return %0 : !hal.buffer_view
// }

// // -----

// util.func public @aliased_storage(%arg0: !hal.buffer_view, %arg1: !hal.buffer, %arg2: !hal.fence) {
//   %c16 = arith.constant 16 : index
//   %c0 = arith.constant 0 : index
//   %c4 = arith.constant 4 : index
//   %element_type_i32 = hal.element_type<i32> : i32
//   %dense_row_major = hal.encoding_type<dense_row_major> : i32
//   hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("tensor") shape([%c4]) type(%element_type_i32) encoding(%dense_row_major)
//   %0 = stream.tensor.import on(#hal.device.promise<@dev_a>) %arg0 : !hal.buffer_view -> tensor<4xi32> in !stream.resource<external>{%c16}
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with(%0 as %arg3: !stream.resource<external>{%c16}) -> !stream.resource<transient>{%c16} {
//     %2 = stream.async.dispatch @dispatch(%arg3[%c0 to %c16 for %c16]) : (!stream.resource<external>{%c16}) -> !stream.resource<transient>{%c16}
//     stream.yield %2 : !stream.resource<transient>{%c16}
//   } => !stream.timepoint
//   %1 = stream.tensor.import on(#hal.device.promise<@dev_b>) %arg1 : !hal.buffer -> tensor<4xi32> in !stream.resource<external>{%c16}
//   %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_b>) await(%result_timepoint) => with(%1 as %arg3: !stream.resource<external>{%c16}, %results as %arg4: !stream.resource<transient>{%c16}) -> %1{%c16} {
//     %2 = stream.async.update %arg4, %arg3[%c0 to %c16] : !stream.resource<transient>{%c16} -> %arg3 as !stream.resource<external>{%c16}
//     stream.yield %2 : !stream.resource<external>{%c16}
//   } => !stream.timepoint
//   stream.timepoint.chain_external on(#hal.device.promise<@dev_b>) %result_timepoint_1 => (%arg2 : !hal.fence)
//   util.return
// }

// // -----

// util.func public @tied_aliased_storage(%arg0: !hal.buffer_view, %arg1: !hal.buffer, %arg2: !hal.fence) {
//   %c0 = arith.constant 0 : index
//   %c123_i32 = arith.constant 123 : i32
//   %c16 = arith.constant 16 : index
//   %0 = stream.tensor.import on(#hal.device.promise<@dev_b>) %arg1 : !hal.buffer -> tensor<4xi32> in !stream.resource<external>{%c16}
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_b>) with(%0 as %arg3: !stream.resource<external>{%c16}) -> %0{%c16} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c16}
//     %2 = stream.async.dispatch @dispatch0(%1[%c0 to %c16 for %c16]) : (!stream.resource<transient>{%c16}) -> !stream.resource<external>{%c16}
//     %3 = stream.async.dispatch @dispatch1(%2[%c0 to %c16 for %c16]) : (!stream.resource<external>{%c16}) -> %2{%c16}
//     %4 = stream.async.update %3, %arg3[%c0 to %c16] : !stream.resource<external>{%c16} -> %arg3 as !stream.resource<external>{%c16}
//     stream.yield %4 : !stream.resource<external>{%c16}
//   } => !stream.timepoint
//   stream.timepoint.chain_external on(#hal.device.promise<@dev_b>) %result_timepoint => (%arg2 : !hal.fence)
//   util.return
// }

// // -----

// util.func public @tied_constant() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     %2 = stream.async.dispatch @a(%1[%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}) -> %1{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   util.return %0, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @tied_constant_multi_consumer() -> (!stream.resource<external>, index, !stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results:2, %result_timepoint = stream.async.execute with() -> (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     %2 = stream.async.dispatch @a(%1[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%c4}) -> %1{%c4}
//     %3:2 = stream.async.concurrent with(%2 as %arg0: !stream.resource<transient>{%c4}) -> (!stream.resource<external>{%c4}, !stream.resource<transient>{%c4}) {
//       %6 = stream.async.transfer %arg0 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_a>) !stream.resource<external>{%c4}
//       %7 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//       stream.yield %6, %7 : !stream.resource<external>{%c4}, !stream.resource<transient>{%c4}
//     }
//     %4 = stream.async.dispatch @b(%3#1[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%c4}) -> %3#1{%c4}
//     %5 = stream.async.transfer %4 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_b>) !stream.resource<external>{%c4}
//     stream.yield %3#0, %5 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0:2 = stream.timepoint.await %result_timepoint => %results#0, %results#1 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//   util.return %0#0, %c4, %0#1, %c4 : !stream.resource<external>, index, !stream.resource<external>, index
// }

// // -----

// util.func public @tied_transfer_constant_multi_consumer() -> (!stream.resource<external>, index, !stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results:2, %result_timepoint = stream.async.execute with() -> (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) {
//     %2 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     %3:2 = stream.async.concurrent with(%2 as %arg0: !stream.resource<transient>{%c4}) -> (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) {
//       %4 = stream.async.transfer %arg0 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_a>) !stream.resource<external>{%c4}
//       %5 = stream.async.transfer %arg0 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_b>) !stream.resource<external>{%c4}
//       stream.yield %4, %5 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//     }
//     stream.yield %3#0, %3#1 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_b>) await(%result_timepoint) => with(%results#1 as %arg0: !stream.resource<external>{%c4}) -> %results#1{%c4} {
//     %2 = stream.async.dispatch @b(%arg0[%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}) -> %arg0{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %results_2, %result_timepoint_3 = stream.async.execute on(#hal.device.promise<@dev_a>) await(%result_timepoint) => with(%results#0 as %arg0: !stream.resource<external>{%c4}) -> %results#0{%c4} {
//     %2 = stream.async.dispatch @a(%arg0[%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}) -> %arg0{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint_3 => %results_2 : !stream.resource<external>{%c4}
//   %1 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//   util.return %0, %c4, %1, %c4 : !stream.resource<external>, index, !stream.resource<external>, index
// }

// // -----

// util.func public @transfer_execution_affinity() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<transient>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     %2 = stream.async.transfer %1 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_b>) !stream.resource<transient>{%c4}
//     stream.yield %2 : !stream.resource<transient>{%c4}
//   } => !stream.timepoint
//   %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_b>) await(%result_timepoint) => with(%results as %arg0: !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4} {
//     %1 = stream.async.dispatch @dispatch(%arg0[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4}
//     stream.yield %1 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//   util.return %0, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @explicit_execution_affinity() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<transient>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     %2 = stream.async.transfer %1 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_b>) !stream.resource<transient>{%c4}
//     stream.yield %2 : !stream.resource<transient>{%c4}
//   } => !stream.timepoint
//   %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_b>) await(%result_timepoint) => with(%results as %arg0: !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4} {
//     %1 = stream.async.dispatch @dispatch(%arg0[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4}
//     stream.yield %1 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//   util.return %0, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @consume_multi_affinities() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c456_i32 = arith.constant 456 : i32
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<transient>{%c4} {
//     %2 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     stream.yield %2 : !stream.resource<transient>{%c4}
//   } => !stream.timepoint
//   %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_b>) with() -> !stream.resource<transient>{%c4} {
//     %2 = stream.async.splat %c456_i32 : i32 -> !stream.resource<transient>{%c4}
//     stream.yield %2 : !stream.resource<transient>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.join max(%result_timepoint, %result_timepoint_1) => !stream.timepoint
//   %results_2, %result_timepoint_3 = stream.async.execute await(%0) => with(%results as %arg0: !stream.resource<transient>{%c4}, %results_0 as %arg1: !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4} {
//     %2 = stream.async.dispatch @dispatch(%arg0[%c0 to %c4 for %c4], %arg1[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%c4}, !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %1 = stream.timepoint.await %result_timepoint_3 => %results_2 : !stream.resource<external>{%c4}
//   util.return %1, %c4 : !stream.resource<external>, index
// }

// // -----

// util.global private @consumed_global_a : !stream.resource<constant>
// util.global private @consumed_global_a_size : index
// util.func public @consumer_fn() -> (!stream.resource<external>, index) {
//   %consumed_global_a = util.global.load immutable @consumed_global_a : !stream.resource<constant>
//   %consumed_global_a_size = util.global.load immutable @consumed_global_a_size : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with(%consumed_global_a as %arg0: !stream.resource<constant>{%consumed_global_a_size}) -> !stream.resource<external>{%consumed_global_a_size} {
//     %1 = stream.async.transfer %arg0 : !stream.resource<constant>{%consumed_global_a_size} -> !stream.resource<external>{%consumed_global_a_size}
//     stream.yield %1 : !stream.resource<external>{%consumed_global_a_size}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%consumed_global_a_size}
//   util.return %0, %consumed_global_a_size : !stream.resource<external>, index
// }

// // -----

// util.global private @consumed_global_ab : !stream.resource<constant>
// util.global private @consumed_global_ab__size : index
// util.func public @consumer_fn_a() -> (!stream.resource<external>, index) {
//   %consumed_global_ab = util.global.load immutable @consumed_global_ab : !stream.resource<constant>
//   %consumed_global_ab__size = util.global.load immutable @consumed_global_ab__size : index
//   %results, %result_timepoint = stream.async.execute with(%consumed_global_ab as %arg0: !stream.resource<constant>{%consumed_global_ab__size}) -> !stream.resource<external>{%consumed_global_ab__size} {
//     %1 = stream.async.transfer %arg0 : !stream.resource<constant>{%consumed_global_ab__size} -> to(#hal.device.promise<@dev_a>) !stream.resource<external>{%consumed_global_ab__size}
//     stream.yield %1 : !stream.resource<external>{%consumed_global_ab__size}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%consumed_global_ab__size}
//   util.return %0, %consumed_global_ab__size : !stream.resource<external>, index
// }
// util.func public @consumer_fn_b() -> (!stream.resource<external>, index) {
//   %consumed_global_ab = util.global.load immutable @consumed_global_ab : !stream.resource<constant>
//   %consumed_global_ab__size = util.global.load immutable @consumed_global_ab__size : index
//   %results, %result_timepoint = stream.async.execute with(%consumed_global_ab as %arg0: !stream.resource<constant>{%consumed_global_ab__size}) -> !stream.resource<external>{%consumed_global_ab__size} {
//     %1 = stream.async.transfer %arg0 : !stream.resource<constant>{%consumed_global_ab__size} -> to(#hal.device.promise<@dev_b>) !stream.resource<external>{%consumed_global_ab__size}
//     stream.yield %1 : !stream.resource<external>{%consumed_global_ab__size}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%consumed_global_ab__size}
//   util.return %0, %consumed_global_ab__size : !stream.resource<external>, index
// }

// // -----

// util.func public @producer_fn() {
//   util.return
// }
// util.func public @consumer_fn() -> (!stream.resource<external>, index) {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_b>) with() -> !stream.resource<external>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %1 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   util.return %0, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @producer_fn() {
//   util.return
// }

// // -----

// util.global private @global_a : !stream.resource<constant>
// util.global private @global_a__size : index
// util.global private @global_b : !stream.resource<constant>
// util.global private @global_b__size : index
// util.func public @consumer_fn() -> (!stream.resource<external>, index) {
//   %c4 = arith.constant 4 : index
//   %c0 = arith.constant 0 : index
//   %global_a = util.global.load immutable @global_a : !stream.resource<constant>
//   %global_a__size = util.global.load immutable @global_a__size : index
//   %global_b = util.global.load immutable @global_b : !stream.resource<constant>
//   %global_b__size = util.global.load immutable @global_b__size : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_b>) with(%global_b as %arg0: !stream.resource<constant>{%global_b__size}, %global_a as %arg1: !stream.resource<constant>{%global_a__size}) -> !stream.resource<external>{%c4} {
//     %1 = stream.async.dispatch @dispatch(%arg1[%c0 to %global_a__size for %global_a__size], %arg0[%c0 to %global_b__size for %global_b__size]) : (!stream.resource<constant>{%global_a__size}, !stream.resource<constant>{%global_b__size}) -> !stream.resource<external>{%c4}
//     stream.yield %1 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   util.return %0, %c4 : !stream.resource<external>, index
// }

// // -----

// util.global private mutable @global_a__timepoint = #stream.timepoint<immediate> : !stream.timepoint
// util.global private mutable @global_a : !stream.resource<variable>
// util.global private mutable @global_a__size = 4 : index
// util.initializer {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<variable>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<variable>{%c4}
//     stream.yield %1 : !stream.resource<variable>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await sync %result_timepoint => %results : !stream.resource<variable>{%c4}
//   util.global.store %0, @global_a : !stream.resource<variable>
//   util.return
// }
// util.func public @step(%arg0: !stream.resource<external>, %arg1: index) -> (!stream.resource<external>, index) {
//   %c4 = arith.constant 4 : index
//   %c8 = arith.constant 8 : index
//   %c0 = arith.constant 0 : index
//   %global_a__timepoint = util.global.load @global_a__timepoint : !stream.timepoint
//   %global_a__size = util.global.load @global_a__size : index
//   %global_a = util.global.load @global_a : !stream.resource<variable>
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) await(%global_a__timepoint) => with(%global_a as %arg2: !stream.resource<variable>{%global_a__size}) -> !stream.resource<variable>{%global_a__size} {
//     %1 = stream.async.transfer %arg2 : !stream.resource<variable>{%global_a__size} -> to(#hal.device.promise<@dev_b>) !stream.resource<variable>{%global_a__size}
//     stream.yield %1 : !stream.resource<variable>{%global_a__size}
//   } => !stream.timepoint
//   %results_0:2, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_b>) await(%result_timepoint) => with(%arg0 as %arg2: !stream.resource<external>{%arg1}, %results as %arg3: !stream.resource<variable>{%global_a__size}) -> (!stream.resource<external>{%c8}, !stream.resource<variable>{%c4}) {
//     %1 = stream.async.transfer %arg2 : !stream.resource<external>{%arg1} -> !stream.resource<transient>{%arg1}
//     %2:2 = stream.async.dispatch @dispatch(%arg3[%c0 to %global_a__size for %global_a__size], %1[%c0 to %arg1 for %arg1]) : (!stream.resource<variable>{%global_a__size}, !stream.resource<transient>{%arg1}) -> (!stream.resource<variable>{%c4}, !stream.resource<external>{%c8})
//     %3 = stream.async.transfer %2#0 : !stream.resource<variable>{%c4} -> to(#hal.device.promise<@dev_a>) !stream.resource<variable>{%c4}
//     stream.yield %2#1, %3 : !stream.resource<external>{%c8}, !stream.resource<variable>{%c4}
//   } => !stream.timepoint
//   util.global.store %results_0#1, @global_a : !stream.resource<variable>
//   util.global.store %c4, @global_a__size : index
//   util.global.store %result_timepoint_1, @global_a__timepoint : !stream.timepoint
//   %0:2 = stream.timepoint.await %result_timepoint_1 => %results_0#1, %results_0#0 : !stream.resource<variable>{%c4}, !stream.resource<external>{%c8}
//   util.return %0#1, %c8 : !stream.resource<external>, index
// }

// // -----

// util.func public @select_constants_consumed(%arg0: i1) -> (!stream.resource<external>, index) {
//   %c456_i32 = arith.constant 456 : i32
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results:2, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) {
//     %2:2 = stream.async.concurrent with() -> (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) {
//       %3 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//       %4 = stream.async.splat %c456_i32 : i32 -> !stream.resource<external>{%c4}
//       stream.yield %3, %4 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//     }

//     // XXX retain allocs
//     retain %2#0
//     retain %2#1

//     stream.yield %2#0, %2#1 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0:2 = stream.timepoint.await %result_timepoint => %results#0, %results#1 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}

//   // XXX retain result, release inputs
//   %1 = arith.select %arg0, %0#0, %0#1 : !stream.resource<external>
//   retain %1
//   release %0#0
//   release %0#1

//   util.return %1, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @select_constants_placed(%arg0: i1) -> (!stream.resource<external>, index) {
//   %c456_i32 = arith.constant 456 : i32
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %3 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %3 : !stream.resource<external>{%c4}
//   } => !stream.timepoint

//   // XXX retain result
//   retain %results

//   %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_b>) with() -> !stream.resource<external>{%c4} {
//     %3 = stream.async.splat %c456_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %3 : !stream.resource<external>{%c4}
//   } => !stream.timepoint

//   // XXX retain results
//   retain %results_0

//   %0 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//   %1 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}

//   // XXX retain result, release inputs
//   %2 = arith.select %arg0, %1, %0 : !stream.resource<external>
//   retain %2
//   release %1
//   release %0

//   util.return %2, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @passthrough_caller() -> (!stream.resource<external>, index) {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %1 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   util.return %0, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @consumer_placement_caller() -> (!stream.resource<external>, index) {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %1 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   util.return %0, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @select_caller(%arg0: !stream.resource<external>, %arg1: index, %arg2: i1) -> (!stream.resource<external>, index) {
//   %c4 = arith.constant 4 : index
//   %c123_i32 = arith.constant 123 : i32
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_b>) with() -> !stream.resource<external>{%c4} {
//     %3 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %3 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   %1 = arith.select %arg2, %arg0, %0 : !stream.resource<external>
//   %2 = arith.select %arg2, %arg1, %c4 : index
//   util.return %1, %2 : !stream.resource<external>, index
// }

// // -----

// util.func public @consumer_multi_placement_caller() -> (!stream.resource<external>, index, !stream.resource<external>, index) {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results:2, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_c>) with() -> (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     %2:2 = stream.async.concurrent with(%1 as %arg0: !stream.resource<transient>{%c4}) -> (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) {
//       %3 = stream.async.transfer %arg0 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_a>) !stream.resource<external>{%c4}
//       %4 = stream.async.transfer %arg0 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_b>) !stream.resource<external>{%c4}
//       stream.yield %3, %4 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//     }
//     stream.yield %2#0, %2#1 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0:2 = stream.timepoint.await %result_timepoint => %results#0, %results#1 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//   util.return %0#0, %c4, %0#1, %c4 : !stream.resource<external>, index, !stream.resource<external>, index
// }

// // -----

// util.func public @dispatch_fn_a() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c123_i32 = arith.constant 123 : i32
//   %c16 = arith.constant 16 : index
//   // YYY produces %results
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<transient>{%c16} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c16}
//     %2 = stream.async.dispatch @dispatch_a_0(%1[%c0 to %c16 for %c16]) : (!stream.resource<transient>{%c16}) -> !stream.resource<transient>{%c16}
//     %3 = stream.async.transfer %2 : !stream.resource<transient>{%c16} -> to(#hal.device.promise<@dev_b>) !stream.resource<transient>{%c16}
//     stream.yield %3 : !stream.resource<transient>{%c16}
//   } => !stream.timepoint
//   // YYY consumes %results
//   // YYY produces %results_0
//   %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_b>) await(%result_timepoint) => with(%results as %arg0: !stream.resource<transient>{%c16}) -> !stream.resource<transient>{%c16} {
//     %1 = stream.async.dispatch @dispatch_b(%arg0[%c0 to %c16 for %c16]) : (!stream.resource<transient>{%c16}) -> !stream.resource<transient>{%c16}
//     %2 = stream.async.transfer %1 : !stream.resource<transient>{%c16} -> to(#hal.device.promise<@dev_a>) !stream.resource<transient>{%c16}
//     stream.yield %2 : !stream.resource<transient>{%c16}
//   } => !stream.timepoint
//   // YYY consumes %results_0
//   // YYY produces %results_2
//   %results_2, %result_timepoint_3 = stream.async.execute on(#hal.device.promise<@dev_a>) await(%result_timepoint_1) => with(%results_0 as %arg0: !stream.resource<transient>{%c16}) -> !stream.resource<external>{%c16} {
//     %1 = stream.async.dispatch @dispatch_a_1(%arg0[%c0 to %c16 for %c16]) : (!stream.resource<transient>{%c16}) -> !stream.resource<external>{%c16}
//     stream.yield %1 : !stream.resource<external>{%c16}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint_3 => %results_2 : !stream.resource<external>{%c16}
//   // YYY consumes %results_2 (aka %0)
//   util.return %0, %c16 : !stream.resource<external>, index
// }
// util.func public @dispatch_fn_b(%arg0: !stream.resource<external>, %arg1: index) -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c16 = arith.constant 16 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_b>) with(%arg0 as %arg2: !stream.resource<external>{%arg1}) -> !stream.resource<external>{%c16} {
//     %1 = stream.async.transfer %arg2 : !stream.resource<external>{%arg1} -> !stream.resource<transient>{%arg1}
//     %2 = stream.async.dispatch @dispatch_b(%1[%c0 to %arg1 for %arg1]) : (!stream.resource<transient>{%arg1}) -> !stream.resource<external>{%c16}
//     stream.yield %2 : !stream.resource<external>{%c16}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c16}
//   util.return %0, %c16 : !stream.resource<external>, index
// }

// // -----

// util.func public @dispatch_fn_a() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c123_i32 = arith.constant 123 : i32
//   %c16 = arith.constant 16 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<transient>{%c16} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c16}
//     %2 = stream.async.transfer %1 : !stream.resource<transient>{%c16} -> to(#hal.device.promise<@dev_c>) !stream.resource<transient>{%c16}
//     stream.yield %2 : !stream.resource<transient>{%c16}
//   } => !stream.timepoint
//   %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_c>) await(%result_timepoint) => with(%results as %arg0: !stream.resource<transient>{%c16}) -> !stream.resource<external>{%c16} {
//     %1 = stream.async.dispatch @dispatch_c(%arg0[%c0 to %c16 for %c16]) : (!stream.resource<transient>{%c16}) -> !stream.resource<transient>{%c16}
//     %2 = stream.async.transfer %1 : !stream.resource<transient>{%c16} -> to(#hal.device.promise<@dev_a>) !stream.resource<external>{%c16}
//     stream.yield %2 : !stream.resource<external>{%c16}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c16}
//   util.return %0, %c16 : !stream.resource<external>, index
// }
// util.func public @dispatch_fn_b(%arg0: !stream.resource<external>, %arg1: index) -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c16 = arith.constant 16 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_b>) with(%arg0 as %arg2: !stream.resource<external>{%arg1}) -> !stream.resource<external>{%c16} {
//     %1 = stream.async.transfer %arg2 : !stream.resource<external>{%arg1} -> !stream.resource<transient>{%arg1}
//     %2 = stream.async.dispatch @dispatch_b(%1[%c0 to %arg1 for %arg1]) : (!stream.resource<transient>{%arg1}) -> !stream.resource<external>{%c16}
//     stream.yield %2 : !stream.resource<external>{%c16}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c16}
//   util.return %0, %c16 : !stream.resource<external>, index
// }
// util.func public @dispatch_fn_c(%arg0: !stream.resource<external>, %arg1: index) -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c16 = arith.constant 16 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_c>) with(%arg0 as %arg2: !stream.resource<external>{%arg1}) -> !stream.resource<external>{%c16} {
//     %1 = stream.async.transfer %arg2 : !stream.resource<external>{%arg1} -> !stream.resource<transient>{%arg1}
//     %2 = stream.async.dispatch @dispatch_c(%1[%c0 to %arg1 for %arg1]) : (!stream.resource<transient>{%arg1}) -> !stream.resource<external>{%c16}
//     stream.yield %2 : !stream.resource<external>{%c16}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c16}
//   util.return %0, %c16 : !stream.resource<external>, index
// }

// // -----

// util.func public @cfg_branch_constant_consumed() -> (!stream.resource<external>, index) {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %1 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   util.return %0, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @cfg_branch_dispatch_produced() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     %2 = stream.async.dispatch @dispatch(%1[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   util.return %0, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @cfg_cond_branch_select(%arg0: i1) -> (!stream.resource<external>, index) {
//   %c456_i32 = arith.constant 456 : i32
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %3 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %3 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_b>) with() -> !stream.resource<external>{%c4} {
//     %3 = stream.async.splat %c456_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %3 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//   %1 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   %2 = arith.select %arg0, %1, %0 : !stream.resource<external>
//   util.return %2, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @cfg_cond_branch_select_consumer(%arg0: i1) -> (!stream.resource<external>, index) {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute with() -> !stream.resource<transient>{%c4} {
//     %2 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     stream.yield %2 : !stream.resource<transient>{%c4}
//   } => !stream.timepoint
//   cf.cond_br %arg0, ^bb1, ^bb2
// ^bb1:  // pred: ^bb0
//   %results_0, %result_timepoint_1 = stream.async.execute await(%result_timepoint) => with(%results as %arg1: !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4} {
//     %2 = stream.async.transfer %arg1 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_a>) !stream.resource<external>{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//   util.return %0, %c4 : !stream.resource<external>, index
// ^bb2:  // pred: ^bb0
//   %results_2, %result_timepoint_3 = stream.async.execute await(%result_timepoint) => with(%results as %arg1: !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4} {
//     %2 = stream.async.transfer %arg1 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_b>) !stream.resource<external>{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %1 = stream.timepoint.await %result_timepoint_3 => %results_2 : !stream.resource<external>{%c4}
//   util.return %1, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @scf_if_capture_consumer(%arg0: i1) -> (!stream.resource<external>, index) {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute with() -> !stream.resource<transient>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     stream.yield %1 : !stream.resource<transient>{%c4}
//   } => !stream.timepoint
//   %0 = scf.if %arg0 -> (!stream.resource<external>) {
//     %results_0, %result_timepoint_1 = stream.async.execute await(%result_timepoint) => with(%results as %arg1: !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4} {
//       %2 = stream.async.transfer %arg1 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_a>) !stream.resource<external>{%c4}
//       stream.yield %2 : !stream.resource<external>{%c4}
//     } => !stream.timepoint
//     %1 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//     scf.yield %1 : !stream.resource<external>
//   } else {
//     %results_0, %result_timepoint_1 = stream.async.execute await(%result_timepoint) => with(%results as %arg1: !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4} {
//       %2 = stream.async.transfer %arg1 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_b>) !stream.resource<external>{%c4}
//       stream.yield %2 : !stream.resource<external>{%c4}
//     } => !stream.timepoint
//     %1 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//     scf.yield %1 : !stream.resource<external>
//   }
//   util.return %0, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @scf_if_capture_producer(%arg0: i1) -> (!stream.resource<external>, index) {
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<transient>{%c4} {
//     %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     stream.yield %1 : !stream.resource<transient>{%c4}
//   } => !stream.timepoint
//   %0 = scf.if %arg0 -> (!stream.resource<external>) {
//     %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_a>) await(%result_timepoint) => with(%results as %arg1: !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4} {
//       %2 = stream.async.transfer %arg1 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_b>) !stream.resource<external>{%c4}
//       stream.yield %2 : !stream.resource<external>{%c4}
//     } => !stream.timepoint
//     %1 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//     scf.yield %1 : !stream.resource<external>
//   } else {
//     %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_a>) await(%result_timepoint) => with(%results as %arg1: !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4} {
//       %2 = stream.async.transfer %arg1 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_c>) !stream.resource<external>{%c4}
//       stream.yield %2 : !stream.resource<external>{%c4}
//     } => !stream.timepoint
//     %1 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//     scf.yield %1 : !stream.resource<external>
//   }
//   util.return %0, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @scf_if_consumer_yield(%arg0: i1) -> (!stream.resource<external>, index) {
//   %c123_i32 = arith.constant 123 : i32
//   %c456_i32 = arith.constant 456 : i32
//   %c4 = arith.constant 4 : index
//   %results:2, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) {
//     %2:2 = stream.async.concurrent with() -> (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) {
//       %3 = stream.async.splat %c456_i32 : i32 -> !stream.resource<external>{%c4}
//       %4 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//       stream.yield %3, %4 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//     }
//     stream.yield %2#0, %2#1 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0:2 = stream.timepoint.await %result_timepoint => %results#0, %results#1 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
//   %1 = arith.select %arg0, %0#1, %0#0 : !stream.resource<external>
//   util.return %1, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @scf_for_consumer_body_transfer() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c2 = arith.constant 2 : index
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %2 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   %1 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %0) -> (!stream.resource<external>) {
//     %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_a>) with(%arg1 as %arg2: !stream.resource<external>{%c4}) -> !stream.resource<external>{%c4} {
//       %3 = stream.async.dispatch @dispatch(%arg2[%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}) -> !stream.resource<external>{%c4}
//       stream.yield %3 : !stream.resource<external>{%c4}
//     } => !stream.timepoint
//     %2 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//     scf.yield %2 : !stream.resource<external>
//   }
//   util.return %1, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @scf_for_boundary_transfer() -> (!stream.resource<external>, index, !stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c2 = arith.constant 2 : index
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results:2, %result_timepoint = stream.async.execute with() -> (!stream.resource<transient>{%c4}, !stream.resource<external>{%c4}) {
//     %3 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     %4 = stream.async.transfer %3 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_b>) !stream.resource<external>{%c4}
//     stream.yield %3, %4 : !stream.resource<transient>{%c4}, !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0:2 = stream.timepoint.await %result_timepoint => %results#0, %results#1 : !stream.resource<transient>{%c4}, !stream.resource<external>{%c4}
//   %1 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %0#0) -> (!stream.resource<transient>) {
//     %results_2, %result_timepoint_3 = stream.async.execute on(#hal.device.promise<@dev_a>) with(%arg1 as %arg2: !stream.resource<transient>{%c4}) -> !stream.resource<transient>{%c4} {
//       %4 = stream.async.dispatch @dispatch(%arg2[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%c4}) -> !stream.resource<transient>{%c4}
//       stream.yield %4 : !stream.resource<transient>{%c4}
//     } => !stream.timepoint
//     %3 = stream.timepoint.await %result_timepoint_3 => %results_2 : !stream.resource<transient>{%c4}
//     scf.yield %3 : !stream.resource<transient>
//   }
//   %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_a>) with(%1 as %arg0: !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4} {
//     %3 = stream.async.transfer %arg0 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_b>) !stream.resource<external>{%c4}
//     stream.yield %3 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %2 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//   util.return %2, %c4, %0#1, %c4 : !stream.resource<external>, index, !stream.resource<external>, index
// }

// // -----

// util.func public @scf_for_body_transfer() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c2 = arith.constant 2 : index
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<transient>{%c4} {
//     %3 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     stream.yield %3 : !stream.resource<transient>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<transient>{%c4}
//   %1 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %0) -> (!stream.resource<transient>) {
//     %results_2, %result_timepoint_3 = stream.async.execute on(#hal.device.promise<@dev_b>) with(%arg1 as %arg2: !stream.resource<transient>{%c4}) -> !stream.resource<transient>{%c4} {
//       %4 = stream.async.dispatch @dispatch(%arg2[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%c4}) -> !stream.resource<transient>{%c4}
//       stream.yield %4 : !stream.resource<transient>{%c4}
//     } => !stream.timepoint
//     %3 = stream.timepoint.await %result_timepoint_3 => %results_2 : !stream.resource<transient>{%c4}
//     scf.yield %3 : !stream.resource<transient>
//   }
//   %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_b>) with(%1 as %arg0: !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4} {
//     %3 = stream.async.transfer %arg0 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_c>) !stream.resource<external>{%c4}
//     stream.yield %3 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %2 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//   util.return %2, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @scf_for_capture_producer() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c2 = arith.constant 2 : index
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %2 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   %1 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %0) -> (!stream.resource<external>) {
//     %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_a>) with(%arg1 as %arg2: !stream.resource<external>{%c4}) -> !stream.resource<external>{%c4} {
//       %3 = stream.async.dispatch @dispatch(%arg2[%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}) -> !stream.resource<external>{%c4}
//       stream.yield %3 : !stream.resource<external>{%c4}
//     } => !stream.timepoint
//     %2 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//     scf.yield %2 : !stream.resource<external>
//   }
//   util.return %1, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @scf_while_consumer_body_transfer() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c2_i32 = arith.constant 2 : i32
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute with() -> !stream.resource<external>{%c4} {
//     %2 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   %1 = scf.while (%arg0 = %0) : (!stream.resource<external>) -> !stream.resource<external> {
//     %results_0, %result_timepoint_1 = stream.async.execute with(%arg0 as %arg1: !stream.resource<external>{%c4}) -> !stream.resource<staging>{%c4} {
//       %5 = stream.async.transfer %arg1 : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
//       stream.yield %5 : !stream.resource<staging>{%c4}
//     } => !stream.timepoint
//     %2 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<staging>{%c4}
//     %3 = stream.async.load %2[%c0] : !stream.resource<staging>{%c4} -> i32
//     %4 = arith.cmpi slt, %3, %c2_i32 : i32
//     scf.condition(%4) %arg0 : !stream.resource<external>
//   } do {
//   ^bb0(%arg0: !stream.resource<external>):
//     %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_a>) with(%arg0 as %arg1: !stream.resource<external>{%c4}) -> !stream.resource<external>{%c4} {
//       %3 = stream.async.transfer %arg1 : !stream.resource<external>{%c4} -> !stream.resource<transient>{%c4}
//       %4 = stream.async.dispatch @dispatch(%3[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4}
//       stream.yield %4 : !stream.resource<external>{%c4}
//     } => !stream.timepoint
//     %2 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//     scf.yield %2 : !stream.resource<external>
//   }
//   util.return %1, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @scf_while_consumer_result_transfer() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c2_i32 = arith.constant 2 : i32
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %2 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   %1 = scf.while (%arg0 = %0) : (!stream.resource<external>) -> !stream.resource<external> {
//     %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_a>) with(%arg0 as %arg1: !stream.resource<external>{%c4}) -> !stream.resource<staging>{%c4} {
//       %5 = stream.async.transfer %arg1 : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
//       stream.yield %5 : !stream.resource<staging>{%c4}
//     } => !stream.timepoint
//     %2 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<staging>{%c4}
//     %3 = stream.async.load %2[%c0] : !stream.resource<staging>{%c4} -> i32
//     %4 = arith.cmpi slt, %3, %c2_i32 : i32
//     scf.condition(%4) %arg0 : !stream.resource<external>
//   } do {
//   ^bb0(%arg0: !stream.resource<external>):
//     %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_a>) with(%arg0 as %arg1: !stream.resource<external>{%c4}) -> !stream.resource<external>{%c4} {
//       %3 = stream.async.dispatch @dispatch(%arg1[%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}) -> !stream.resource<external>{%c4}
//       stream.yield %3 : !stream.resource<external>{%c4}
//     } => !stream.timepoint
//     %2 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//     scf.yield %2 : !stream.resource<external>
//   }
//   util.return %1, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @scf_while_body_transfer() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c2_i32 = arith.constant 2 : i32
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<transient>{%c4} {
//     %3 = stream.async.splat %c123_i32 : i32 -> !stream.resource<transient>{%c4}
//     stream.yield %3 : !stream.resource<transient>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<transient>{%c4}
//   %1 = scf.while (%arg0 = %0) : (!stream.resource<transient>) -> !stream.resource<transient> {
//     %results_2, %result_timepoint_3 = stream.async.execute with(%arg0 as %arg1: !stream.resource<transient>{%c4}) -> !stream.resource<staging>{%c4} {
//       %6 = stream.async.transfer %arg1 : !stream.resource<transient>{%c4} -> !stream.resource<staging>{%c4}
//       stream.yield %6 : !stream.resource<staging>{%c4}
//     } => !stream.timepoint
//     %3 = stream.timepoint.await %result_timepoint_3 => %results_2 : !stream.resource<staging>{%c4}
//     %4 = stream.async.load %3[%c0] : !stream.resource<staging>{%c4} -> i32
//     %5 = arith.cmpi slt, %4, %c2_i32 : i32
//     scf.condition(%5) %arg0 : !stream.resource<transient>
//   } do {
//   ^bb0(%arg0: !stream.resource<transient>):
//     %results_2, %result_timepoint_3 = stream.async.execute on(#hal.device.promise<@dev_b>) with(%arg0 as %arg1: !stream.resource<transient>{%c4}) -> !stream.resource<transient>{%c4} {
//       %4 = stream.async.dispatch @dispatch(%arg1[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%c4}) -> !stream.resource<transient>{%c4}
//       stream.yield %4 : !stream.resource<transient>{%c4}
//     } => !stream.timepoint
//     %3 = stream.timepoint.await %result_timepoint_3 => %results_2 : !stream.resource<transient>{%c4}
//     scf.yield %3 : !stream.resource<transient>
//   }
//   %results_0, %result_timepoint_1 = stream.async.execute with(%1 as %arg0: !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4} {
//     %3 = stream.async.transfer %arg0 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_c>) !stream.resource<external>{%c4}
//     stream.yield %3 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %2 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//   util.return %2, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @scf_while_capture_producer_condition() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c2_i32 = arith.constant 2 : i32
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %2 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   %1 = scf.while (%arg0 = %0) : (!stream.resource<external>) -> !stream.resource<external> {
//     %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_a>) with(%arg0 as %arg1: !stream.resource<external>{%c4}) -> !stream.resource<staging>{%c4} {
//       %5 = stream.async.transfer %arg1 : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
//       stream.yield %5 : !stream.resource<staging>{%c4}
//     } => !stream.timepoint
//     %2 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<staging>{%c4}
//     %3 = stream.async.load %2[%c0] : !stream.resource<staging>{%c4} -> i32
//     %4 = arith.cmpi slt, %3, %c2_i32 : i32
//     scf.condition(%4) %arg0 : !stream.resource<external>
//   } do {
//   ^bb0(%arg0: !stream.resource<external>):
//     %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_a>) with(%arg0 as %arg1: !stream.resource<external>{%c4}) -> !stream.resource<external>{%c4} {
//       %3 = stream.async.dispatch @dispatch(%arg1[%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}) -> !stream.resource<external>{%c4}
//       stream.yield %3 : !stream.resource<external>{%c4}
//     } => !stream.timepoint
//     %2 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//     scf.yield %2 : !stream.resource<external>
//   }
//   util.return %1, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @scf_while_capture_producer_body() -> (!stream.resource<external>, index) {
//   %c0 = arith.constant 0 : index
//   %c2_i32 = arith.constant 2 : i32
//   %c123_i32 = arith.constant 123 : i32
//   %c4 = arith.constant 4 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) with() -> !stream.resource<external>{%c4} {
//     %2 = stream.async.splat %c123_i32 : i32 -> !stream.resource<external>{%c4}
//     stream.yield %2 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c4}
//   %1 = scf.while (%arg0 = %0) : (!stream.resource<external>) -> !stream.resource<external> {
//     %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_a>) with(%arg0 as %arg1: !stream.resource<external>{%c4}) -> !stream.resource<staging>{%c4} {
//       %5 = stream.async.transfer %arg1 : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
//       stream.yield %5 : !stream.resource<staging>{%c4}
//     } => !stream.timepoint
//     %2 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<staging>{%c4}
//     %3 = stream.async.load %2[%c0] : !stream.resource<staging>{%c4} -> i32
//     %4 = arith.cmpi slt, %3, %c2_i32 : i32
//     scf.condition(%4) %arg0 : !stream.resource<external>
//   } do {
//   ^bb0(%arg0: !stream.resource<external>):
//     %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_a>) with(%arg0 as %arg1: !stream.resource<external>{%c4}) -> !stream.resource<external>{%c4} {
//       %3 = stream.async.dispatch @dispatch(%arg1[%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}) -> !stream.resource<external>{%c4}
//       stream.yield %3 : !stream.resource<external>{%c4}
//     } => !stream.timepoint
//     %2 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
//     scf.yield %2 : !stream.resource<external>
//   }
//   util.return %1, %c4 : !stream.resource<external>, index
// }

// // -----

// util.func public @simple_program(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view {
//   %c0 = arith.constant 0 : index
//   %c4 = arith.constant 4 : index
//   %c1_i32 = arith.constant 1 : i32
//   %c2_i32 = arith.constant 2 : i32
//   %c1 = arith.constant 1 : index
//   %element_type_i32 = hal.element_type<i32> : i32
//   %dense_row_major = hal.encoding_type<dense_row_major> : i32
//   hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c1]) type(%element_type_i32) encoding(%dense_row_major)
//   %0 = stream.tensor.import on(#hal.device.promise<@dev_a>) %arg0 : !hal.buffer_view -> tensor<1xi32> in !stream.resource<external>{%c4}
//   %1 = stream.timepoint.import on(#hal.device.promise<@dev_a>) %arg1 : (!hal.fence) => !stream.timepoint
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) await(%1) => with(%0 as %arg3: !stream.resource<external>{%c4}) -> !stream.resource<transient>{%c4} {
//     %3 = stream.async.splat %c1_i32 : i32 -> !stream.resource<transient>{%c4}
//     %4 = stream.async.dispatch @dispatch_a(%arg3[%c0 to %c4 for %c4], %3[%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}, !stream.resource<transient>{%c4}) -> !stream.resource<transient>{%c4}
//     %5 = stream.async.transfer %4 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_b>) !stream.resource<transient>{%c4}
//     stream.yield %5 : !stream.resource<transient>{%c4}
//   } => !stream.timepoint
//   %results_0, %result_timepoint_1 = stream.async.execute on(#hal.device.promise<@dev_b>) await(%result_timepoint) => with(%results as %arg3: !stream.resource<transient>{%c4}) -> !stream.resource<external>{%c4} {
//     %3 = stream.async.splat %c2_i32 : i32 -> !stream.resource<transient>{%c4}
//     %4 = stream.async.dispatch @dispatch_b(%arg3[%c0 to %c4 for %c4], %3[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%c4}, !stream.resource<transient>{%c4}) -> !stream.resource<transient>{%c4}
//     %5 = stream.async.transfer %4 : !stream.resource<transient>{%c4} -> to(#hal.device.promise<@dev_a>) !stream.resource<external>{%c4}
//     stream.yield %5 : !stream.resource<external>{%c4}
//   } => !stream.timepoint
//   stream.timepoint.chain_external on(#hal.device.promise<@dev_a>) %result_timepoint_1 => (%arg2 : !hal.fence)
//   %2 = stream.tensor.export on(#hal.device.promise<@dev_a>) %results_0 : tensor<1xi32> in !stream.resource<external>{%c4} -> !hal.buffer_view
//   util.return %2 : !hal.buffer_view
// }
// util.func private @dispatch_a(%arg0: !stream.resource<transient>, %arg1: !stream.timepoint, %arg2: index) -> (!stream.resource<transient>, !stream.timepoint) {
//   %c4 = arith.constant 4 : index
//   %c1_i32 = arith.constant 1 : i32
//   %c0 = arith.constant 0 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_a>) await(%arg1) => with(%arg0 as %arg3: !stream.resource<transient>{%arg2}) -> !stream.resource<transient>{%c4} {
//     %0 = stream.async.splat %c1_i32 : i32 -> !stream.resource<transient>{%c4}
//     %1 = stream.async.dispatch @dispatch_a(%arg3[%c0 to %arg2 for %arg2], %0[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%arg2}, !stream.resource<transient>{%c4}) -> !stream.resource<transient>{%c4}
//     stream.yield %1 : !stream.resource<transient>{%c4}
//   } => !stream.timepoint
//   util.return %results, %result_timepoint : !stream.resource<transient>, !stream.timepoint
// }
// util.func private @dispatch_b(%arg0: !stream.resource<transient>, %arg1: !stream.timepoint, %arg2: index) -> (!stream.resource<transient>, !stream.timepoint) {
//   %c4 = arith.constant 4 : index
//   %c2_i32 = arith.constant 2 : i32
//   %c0 = arith.constant 0 : index
//   %results, %result_timepoint = stream.async.execute on(#hal.device.promise<@dev_b>) await(%arg1) => with(%arg0 as %arg3: !stream.resource<transient>{%arg2}) -> !stream.resource<transient>{%c4} {
//     %0 = stream.async.splat %c2_i32 : i32 -> !stream.resource<transient>{%c4}
//     %1 = stream.async.dispatch @dispatch_b(%arg3[%c0 to %arg2 for %arg2], %0[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%arg2}, !stream.resource<transient>{%c4}) -> !stream.resource<transient>{%c4}
//     stream.yield %1 : !stream.resource<transient>{%c4}
//   } => !stream.timepoint
//   util.return %results, %result_timepoint : !stream.resource<transient>, !stream.timepoint
// }
