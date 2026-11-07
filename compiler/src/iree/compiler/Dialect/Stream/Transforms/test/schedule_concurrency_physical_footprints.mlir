// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-stream-schedule-concurrency)" %s | FileCheck %s

// The fill at byte 6 patches the word holding bytes 4-7, which the update
// writes: they may not share a concurrent wave.

// CHECK-LABEL: @physical_fill_copy_hazard
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.update
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.fill
util.func public @physical_fill_copy_hazard(%input: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">,
                stream.resources = #stream.resource_config<{max_allocation_size = 4096, min_buffer_offset_alignment = 64, max_buffer_range = 2048, min_buffer_range_alignment = 64, index_bits = 32, min_fill_granularity = 4}>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  %c0_i8 = arith.constant 0 : i8
  %results, %timepoint = stream.async.execute with(%input as %captured: !stream.resource<external>{%c6}) -> !stream.resource<external>{%c12} {
    %dispatched = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%captured[%c0 to %c6 for %c6]) : (!stream.resource<external>{%c6}) -> !stream.resource<transient>{%c6}
    %target = stream.async.alloca : !stream.resource<external>{%c12}
    %updated = stream.async.update %dispatched, %target[%c0 to %c6] : !stream.resource<transient>{%c6} -> %target as !stream.resource<external>{%c12}
    %filled = stream.async.fill %c0_i8, %updated[%c6 to %c12 for %c6] : i8 -> %updated as !stream.resource<external>{%c12}
    stream.yield %filled : !stream.resource<external>{%c12}
  } => !stream.timepoint
  %0 = stream.timepoint.await %timepoint => %results : !stream.resource<external>{%c12}
  util.return %0 : !stream.resource<external>
}

// -----

// A boundary at byte 8 splits whole words: the two stay concurrent.

// CHECK-LABEL: @physical_fill_aligned_disjoint
//       CHECK:   stream.async.concurrent
//  CHECK-NEXT:     stream.async.update
//  CHECK-NEXT:     stream.async.fill
util.func public @physical_fill_aligned_disjoint(%input: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">,
                stream.resources = #stream.resource_config<{max_allocation_size = 4096, min_buffer_offset_alignment = 64, max_buffer_range = 2048, min_buffer_range_alignment = 64, index_bits = 32, min_fill_granularity = 4}>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c0_i8 = arith.constant 0 : i8
  %results, %timepoint = stream.async.execute with(%input as %captured: !stream.resource<external>{%c8}) -> !stream.resource<external>{%c16} {
    %dispatched = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%captured[%c0 to %c8 for %c8]) : (!stream.resource<external>{%c8}) -> !stream.resource<transient>{%c8}
    %target = stream.async.alloca : !stream.resource<external>{%c16}
    %updated = stream.async.update %dispatched, %target[%c0 to %c8] : !stream.resource<transient>{%c8} -> %target as !stream.resource<external>{%c16}
    %filled = stream.async.fill %c0_i8, %updated[%c8 to %c16 for %c8] : i8 -> %updated as !stream.resource<external>{%c16}
    stream.yield %filled : !stream.resource<external>{%c16}
  } => !stream.timepoint
  %0 = stream.timepoint.await %timepoint => %results : !stream.resource<external>{%c16}
  util.return %0 : !stream.resource<external>
}

// -----

// A device that fills bytes keeps a byte boundary concurrent.

// CHECK-LABEL: @physical_fill_byte_granular
//       CHECK:   stream.async.concurrent
//  CHECK-NEXT:     stream.async.update
//  CHECK-NEXT:     stream.async.fill
util.func public @physical_fill_byte_granular(%input: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  %c0_i8 = arith.constant 0 : i8
  %results, %timepoint = stream.async.execute with(%input as %captured: !stream.resource<external>{%c6}) -> !stream.resource<external>{%c12} {
    %dispatched = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%captured[%c0 to %c6 for %c6]) : (!stream.resource<external>{%c6}) -> !stream.resource<transient>{%c6}
    %target = stream.async.alloca : !stream.resource<external>{%c12}
    %updated = stream.async.update %dispatched, %target[%c0 to %c6] : !stream.resource<transient>{%c6} -> %target as !stream.resource<external>{%c12}
    %filled = stream.async.fill %c0_i8, %updated[%c6 to %c12 for %c6] : i8 -> %updated as !stream.resource<external>{%c12}
    stream.yield %filled : !stream.resource<external>{%c12}
  } => !stream.timepoint
  %0 = stream.timepoint.await %timepoint => %results : !stream.resource<external>{%c12}
  util.return %0 : !stream.resource<external>
}

// -----

// A dynamic boundary is ordered unless divisibility proves whole words.

// CHECK-LABEL: @physical_fill_dynamic_boundary
//  CHECK-SAME: %[[SIZE:[a-z0-9]+]]: index
//       CHECK:   stream.async.concurrent
//       CHECK:     stream.async.update {{.+}}[%c0 to %[[SIZE]]]
//       CHECK:   stream.async.concurrent
//  CHECK-NEXT:     stream.async.fill {{.+}}[%[[SIZE]] to
//  CHECK-NEXT:     %[[ALIGNED_UPDATE:.+]] = stream.async.update
//  CHECK-NEXT:     stream.async.fill %c0_i8, %[[ALIGNED_UPDATE]]
util.func public @physical_fill_dynamic_boundary(%input: !stream.resource<external>, %size_arg: index) -> (!stream.resource<external>, !stream.resource<external>)
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">,
                stream.resources = #stream.resource_config<{max_allocation_size = 4096, min_buffer_offset_alignment = 64, max_buffer_range = 2048, min_buffer_range_alignment = 64, index_bits = 32, min_fill_granularity = 4}>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i8 = arith.constant 0 : i8
  %aligned = util.assume.int %size_arg<udiv = 4> : index
  %total = arith.addi %size_arg, %size_arg : index
  %aligned_total = arith.addi %aligned, %aligned : index
  %results:2, %timepoint = stream.async.execute with(%input as %captured: !stream.resource<external>{%size_arg}) -> (!stream.resource<external>{%total}, !stream.resource<external>{%aligned_total}) {
    %d0 = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%captured[%c0 to %size_arg for %size_arg]) : (!stream.resource<external>{%size_arg}) -> !stream.resource<transient>{%size_arg}
    %t0 = stream.async.alloca : !stream.resource<external>{%total}
    %u0 = stream.async.update %d0, %t0[%c0 to %size_arg] : !stream.resource<transient>{%size_arg} -> %t0 as !stream.resource<external>{%total}
    %f0 = stream.async.fill %c0_i8, %u0[%size_arg to %total for %size_arg] : i8 -> %u0 as !stream.resource<external>{%total}
    %d1 = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%captured[%c0 to %aligned for %aligned]) : (!stream.resource<external>{%size_arg}) -> !stream.resource<transient>{%aligned}
    %t1 = stream.async.alloca : !stream.resource<external>{%aligned_total}
    %u1 = stream.async.update %d1, %t1[%c0 to %aligned] : !stream.resource<transient>{%aligned} -> %t1 as !stream.resource<external>{%aligned_total}
    %f1 = stream.async.fill %c0_i8, %u1[%aligned to %aligned_total for %aligned] : i8 -> %u1 as !stream.resource<external>{%aligned_total}
    stream.yield %f0, %f1 : !stream.resource<external>{%total}, !stream.resource<external>{%aligned_total}
  } => !stream.timepoint
  %0:2 = stream.timepoint.await %timepoint => %results#0, %results#1 : !stream.resource<external>{%total}, !stream.resource<external>{%aligned_total}
  util.return %0#0, %0#1 : !stream.resource<external>, !stream.resource<external>
}

// -----

// Updates lower to byte-exact copies: two updates touching at byte 6 stay
// concurrent.

// CHECK-LABEL: @physical_updates_byte_exact
//       CHECK:   stream.async.concurrent
//       CHECK:     stream.async.update {{.+}}[%c0 to %c6]
//  CHECK-NEXT:     stream.async.update {{.+}}[%c6 to %c12]
util.func public @physical_updates_byte_exact(%input: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">,
                stream.resources = #stream.resource_config<{max_allocation_size = 4096, min_buffer_offset_alignment = 64, max_buffer_range = 2048, min_buffer_range_alignment = 64, index_bits = 32, min_fill_granularity = 4}>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  %results, %timepoint = stream.async.execute with(%input as %captured: !stream.resource<external>{%c6}) -> !stream.resource<external>{%c12} {
    %a = stream.async.dispatch @ex::@a[%c1, %c1, %c1](%captured[%c0 to %c6 for %c6]) : (!stream.resource<external>{%c6}) -> !stream.resource<transient>{%c6}
    %b = stream.async.dispatch @ex::@b[%c1, %c1, %c1](%captured[%c0 to %c6 for %c6]) : (!stream.resource<external>{%c6}) -> !stream.resource<transient>{%c6}
    %target = stream.async.alloca : !stream.resource<external>{%c12}
    %u0 = stream.async.update %a, %target[%c0 to %c6] : !stream.resource<transient>{%c6} -> %target as !stream.resource<external>{%c12}
    %u1 = stream.async.update %b, %u0[%c6 to %c12] : !stream.resource<transient>{%c6} -> %u0 as !stream.resource<external>{%c12}
    stream.yield %u1 : !stream.resource<external>{%c12}
  } => !stream.timepoint
  %0 = stream.timepoint.await %timepoint => %results : !stream.resource<external>{%c12}
  util.return %0 : !stream.resource<external>
}

// -----

// The granularity a device target records reaches the hazard analysis through
// its affinity.

util.global private @device = #hal.device.target<"word", {stream.min_fill_granularity = 4 : i64}> : !hal.device

// CHECK-LABEL: @physical_fill_device_affinity
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.update
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.fill
util.func public @physical_fill_device_affinity(%input: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  %c0_i8 = arith.constant 0 : i8
  %results, %timepoint = stream.async.execute on(#hal.device.affinity<@device>) with(%input as %captured: !stream.resource<external>{%c6}) -> !stream.resource<external>{%c12} {
    %dispatched = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%captured[%c0 to %c6 for %c6]) : (!stream.resource<external>{%c6}) -> !stream.resource<transient>{%c6}
    %target = stream.async.alloca : !stream.resource<external>{%c12}
    %updated = stream.async.update %dispatched, %target[%c0 to %c6] : !stream.resource<transient>{%c6} -> %target as !stream.resource<external>{%c12}
    %filled = stream.async.fill %c0_i8, %updated[%c6 to %c12 for %c6] : i8 -> %updated as !stream.resource<external>{%c12}
    stream.yield %filled : !stream.resource<external>{%c12}
  } => !stream.timepoint
  %0 = stream.timepoint.await %timepoint => %results : !stream.resource<external>{%c12}
  util.return %0 : !stream.resource<external>
}

// -----

// A device whose copies patch whole words: the update at byte 6 shares the
// word holding bytes 4-7 with the first.

// CHECK-LABEL: @physical_copy_update_hazard
//       CHECK:   stream.async.update {{.+}}[%c0 to %c6]
//  CHECK-NEXT:   stream.yield
//       CHECK:   stream.async.update {{.+}}[%c6 to %c12]
util.func public @physical_copy_update_hazard(%input: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">,
                stream.resources = #stream.resource_config<{max_allocation_size = 4096, min_buffer_offset_alignment = 64, max_buffer_range = 2048, min_buffer_range_alignment = 64, index_bits = 32, min_copy_granularity = 4}>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  %results, %timepoint = stream.async.execute with(%input as %captured: !stream.resource<external>{%c6}) -> !stream.resource<external>{%c12} {
    %a = stream.async.dispatch @ex::@a[%c1, %c1, %c1](%captured[%c0 to %c6 for %c6]) : (!stream.resource<external>{%c6}) -> !stream.resource<transient>{%c6}
    %b = stream.async.dispatch @ex::@b[%c1, %c1, %c1](%captured[%c0 to %c6 for %c6]) : (!stream.resource<external>{%c6}) -> !stream.resource<transient>{%c6}
    %target = stream.async.alloca : !stream.resource<external>{%c12}
    %u0 = stream.async.update %a, %target[%c0 to %c6] : !stream.resource<transient>{%c6} -> %target as !stream.resource<external>{%c12}
    %u1 = stream.async.update %b, %u0[%c6 to %c12] : !stream.resource<transient>{%c6} -> %u0 as !stream.resource<external>{%c12}
    stream.yield %u1 : !stream.resource<external>{%c12}
  } => !stream.timepoint
  %0 = stream.timepoint.await %timepoint => %results : !stream.resource<external>{%c12}
  util.return %0 : !stream.resource<external>
}

// -----

// The same updates at byte 8 write whole words of their own.

// CHECK-LABEL: @physical_copy_aligned_disjoint
//       CHECK:   stream.async.concurrent
//       CHECK:     stream.async.update {{.+}}[%c0 to %c8]
//  CHECK-NEXT:     stream.async.update {{.+}}[%c8 to %c16]
util.func public @physical_copy_aligned_disjoint(%input: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">,
                stream.resources = #stream.resource_config<{max_allocation_size = 4096, min_buffer_offset_alignment = 64, max_buffer_range = 2048, min_buffer_range_alignment = 64, index_bits = 32, min_copy_granularity = 4}>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %results, %timepoint = stream.async.execute with(%input as %captured: !stream.resource<external>{%c8}) -> !stream.resource<external>{%c16} {
    %a = stream.async.dispatch @ex::@a[%c1, %c1, %c1](%captured[%c0 to %c8 for %c8]) : (!stream.resource<external>{%c8}) -> !stream.resource<transient>{%c8}
    %b = stream.async.dispatch @ex::@b[%c1, %c1, %c1](%captured[%c0 to %c8 for %c8]) : (!stream.resource<external>{%c8}) -> !stream.resource<transient>{%c8}
    %target = stream.async.alloca : !stream.resource<external>{%c16}
    %u0 = stream.async.update %a, %target[%c0 to %c8] : !stream.resource<transient>{%c8} -> %target as !stream.resource<external>{%c16}
    %u1 = stream.async.update %b, %u0[%c8 to %c16] : !stream.resource<transient>{%c8} -> %u0 as !stream.resource<external>{%c16}
    stream.yield %u1 : !stream.resource<external>{%c16}
  } => !stream.timepoint
  %0 = stream.timepoint.await %timepoint => %results : !stream.resource<external>{%c16}
  util.return %0 : !stream.resource<external>
}

// -----

// A dispatch reading bytes 6-11 reads the word the copy into bytes 0-5
// rewrites whole.

// CHECK-LABEL: @physical_copy_dispatch_hazard
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.copy
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.dispatch @ex::@reader
util.func public @physical_copy_dispatch_hazard(%input: !stream.resource<external>) -> (!stream.resource<external>, !stream.resource<external>)
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">,
                stream.resources = #stream.resource_config<{max_allocation_size = 4096, min_buffer_offset_alignment = 64, max_buffer_range = 2048, min_buffer_range_alignment = 64, index_bits = 32, min_copy_granularity = 4}>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  %results:2, %timepoint = stream.async.execute with(%input as %captured: !stream.resource<external>{%c12}) -> (!stream.resource<external>{%c12}, !stream.resource<external>{%c6}) {
    %a = stream.async.dispatch @ex::@a[%c1, %c1, %c1](%captured[%c0 to %c6 for %c6]) : (!stream.resource<external>{%c12}) -> !stream.resource<transient>{%c12}
    %target = stream.async.alloca : !stream.resource<external>{%c12}
    %copied = stream.async.copy %a[%c0 to %c6], %target[%c0 to %c6], %c6 : !stream.resource<transient>{%c12} -> %target as !stream.resource<external>{%c12}
    %read = stream.async.dispatch @ex::@reader[%c1, %c1, %c1](%copied[%c6 to %c12 for %c6]) : (!stream.resource<external>{%c12}) -> !stream.resource<external>{%c6}
    stream.yield %copied, %read : !stream.resource<external>{%c12}, !stream.resource<external>{%c6}
  } => !stream.timepoint
  %0:2 = stream.timepoint.await %timepoint => %results#0, %results#1 : !stream.resource<external>{%c12}, !stream.resource<external>{%c6}
  util.return %0#0, %0#1 : !stream.resource<external>, !stream.resource<external>
}

// -----

// A copy reading bytes 0-5 reads the word holding bytes 4-7, which the fill
// at byte 6 writes.

// CHECK-LABEL: @physical_copy_source_read_hazard
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.copy
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.fill
util.func public @physical_copy_source_read_hazard(%input: !stream.resource<external>) -> (!stream.resource<external>, !stream.resource<external>)
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">,
                stream.resources = #stream.resource_config<{max_allocation_size = 4096, min_buffer_offset_alignment = 64, max_buffer_range = 2048, min_buffer_range_alignment = 64, index_bits = 32, min_copy_granularity = 4}>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  %c0_i8 = arith.constant 0 : i8
  %results:2, %timepoint = stream.async.execute with(%input as %captured: !stream.resource<external>{%c12}) -> (!stream.resource<external>{%c6}, !stream.resource<external>{%c12}) {
    %source = stream.async.dispatch @ex::@a[%c1, %c1, %c1](%captured[%c0 to %c12 for %c12]) : (!stream.resource<external>{%c12}) -> !stream.resource<external>{%c12}
    %target = stream.async.alloca : !stream.resource<external>{%c6}
    %copied = stream.async.copy %source[%c0 to %c6], %target[%c0 to %c6], %c6 : !stream.resource<external>{%c12} -> %target as !stream.resource<external>{%c6}
    %filled = stream.async.fill %c0_i8, %source[%c6 to %c12 for %c6] : i8 -> %source as !stream.resource<external>{%c12}
    stream.yield %copied, %filled : !stream.resource<external>{%c6}, !stream.resource<external>{%c12}
  } => !stream.timepoint
  %0:2 = stream.timepoint.await %timepoint => %results#0, %results#1 : !stream.resource<external>{%c6}, !stream.resource<external>{%c12}
  util.return %0#0, %0#1 : !stream.resource<external>, !stream.resource<external>
}

// -----

// A device that records only its dispatch unit leaves the function's own
// configuration, with its fill unit, in force.

util.global private @device = #hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device

// CHECK-LABEL: @physical_affinity_keeps_explicit_config
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.update
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.fill
util.func public @physical_affinity_keeps_explicit_config(%input: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">,
                stream.resources = #stream.resource_config<{max_allocation_size = 4096, min_buffer_offset_alignment = 64, max_buffer_range = 2048, min_buffer_range_alignment = 64, index_bits = 32, min_fill_granularity = 4}>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  %c0_i8 = arith.constant 0 : i8
  %results, %timepoint = stream.async.execute on(#hal.device.affinity<@device>) with(%input as %captured: !stream.resource<external>{%c6}) -> !stream.resource<external>{%c12} {
    %dispatched = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%captured[%c0 to %c6 for %c6]) : (!stream.resource<external>{%c6}) -> !stream.resource<transient>{%c6}
    %target = stream.async.alloca : !stream.resource<external>{%c12}
    %updated = stream.async.update %dispatched, %target[%c0 to %c6] : !stream.resource<transient>{%c6} -> %target as !stream.resource<external>{%c12}
    %filled = stream.async.fill %c0_i8, %updated[%c6 to %c12 for %c6] : i8 -> %updated as !stream.resource<external>{%c12}
    stream.yield %filled : !stream.resource<external>{%c12}
  } => !stream.timepoint
  %0 = stream.timepoint.await %timepoint => %results : !stream.resource<external>{%c12}
  util.return %0 : !stream.resource<external>
}

// -----

// An optimal set of one device fills in that device's words.

util.global private @device = #hal.device.target<"word", {stream.min_fill_granularity = 4 : i64}> : !hal.device

// CHECK-LABEL: @physical_fill_optimal_singleton
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.update
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.fill
util.func public @physical_fill_optimal_singleton(%input: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  %c0_i8 = arith.constant 0 : i8
  %results, %timepoint = stream.async.execute on(#hal.device.optimal<[#hal.device.affinity<@device>]>) with(%input as %captured: !stream.resource<external>{%c6}) -> !stream.resource<external>{%c12} {
    %dispatched = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%captured[%c0 to %c6 for %c6]) : (!stream.resource<external>{%c6}) -> !stream.resource<transient>{%c6}
    %target = stream.async.alloca : !stream.resource<external>{%c12}
    %updated = stream.async.update %dispatched, %target[%c0 to %c6] : !stream.resource<transient>{%c6} -> %target as !stream.resource<external>{%c12}
    %filled = stream.async.fill %c0_i8, %updated[%c6 to %c12 for %c6] : i8 -> %updated as !stream.resource<external>{%c12}
    stream.yield %filled : !stream.resource<external>{%c12}
  } => !stream.timepoint
  %0 = stream.timepoint.await %timepoint => %results : !stream.resource<external>{%c12}
  util.return %0 : !stream.resource<external>
}

// -----

// A byte device in the set does not lift the word device's unit.

util.global private @word = #hal.device.target<"word", {stream.min_fill_granularity = 4 : i64}> : !hal.device
util.global private @byte = #hal.device.target<"byte"> : !hal.device

// CHECK-LABEL: @physical_fill_optimal_mixed
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.update
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.fill
util.func public @physical_fill_optimal_mixed(%input: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  %c0_i8 = arith.constant 0 : i8
  %results, %timepoint = stream.async.execute on(#hal.device.optimal<[#hal.device.affinity<@byte>, #hal.device.affinity<@word>]>) with(%input as %captured: !stream.resource<external>{%c6}) -> !stream.resource<external>{%c12} {
    %dispatched = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%captured[%c0 to %c6 for %c6]) : (!stream.resource<external>{%c6}) -> !stream.resource<transient>{%c6}
    %target = stream.async.alloca : !stream.resource<external>{%c12}
    %updated = stream.async.update %dispatched, %target[%c0 to %c6] : !stream.resource<transient>{%c6} -> %target as !stream.resource<external>{%c12}
    %filled = stream.async.fill %c0_i8, %updated[%c6 to %c12 for %c6] : i8 -> %updated as !stream.resource<external>{%c12}
    stream.yield %filled : !stream.resource<external>{%c12}
  } => !stream.timepoint
  %0 = stream.timepoint.await %timepoint => %results : !stream.resource<external>{%c12}
  util.return %0 : !stream.resource<external>
}

// -----

// Member order does not matter.

util.global private @word = #hal.device.target<"word", {stream.min_fill_granularity = 4 : i64}> : !hal.device
util.global private @byte = #hal.device.target<"byte"> : !hal.device

// CHECK-LABEL: @physical_fill_optimal_mixed_swapped
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.update
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.fill
util.func public @physical_fill_optimal_mixed_swapped(%input: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  %c0_i8 = arith.constant 0 : i8
  %results, %timepoint = stream.async.execute on(#hal.device.optimal<[#hal.device.affinity<@word>, #hal.device.affinity<@byte>]>) with(%input as %captured: !stream.resource<external>{%c6}) -> !stream.resource<external>{%c12} {
    %dispatched = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%captured[%c0 to %c6 for %c6]) : (!stream.resource<external>{%c6}) -> !stream.resource<transient>{%c6}
    %target = stream.async.alloca : !stream.resource<external>{%c12}
    %updated = stream.async.update %dispatched, %target[%c0 to %c6] : !stream.resource<transient>{%c6} -> %target as !stream.resource<external>{%c12}
    %filled = stream.async.fill %c0_i8, %updated[%c6 to %c12 for %c6] : i8 -> %updated as !stream.resource<external>{%c12}
    stream.yield %filled : !stream.resource<external>{%c12}
  } => !stream.timepoint
  %0 = stream.timepoint.await %timepoint => %results : !stream.resource<external>{%c12}
  util.return %0 : !stream.resource<external>
}

// -----

// A boundary at byte 8 splits whole words under the set too.

util.global private @word = #hal.device.target<"word", {stream.min_fill_granularity = 4 : i64}> : !hal.device
util.global private @byte = #hal.device.target<"byte"> : !hal.device

// CHECK-LABEL: @physical_fill_optimal_aligned
//       CHECK:   stream.async.concurrent
//  CHECK-NEXT:     stream.async.update
//  CHECK-NEXT:     stream.async.fill
util.func public @physical_fill_optimal_aligned(%input: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c0_i8 = arith.constant 0 : i8
  %results, %timepoint = stream.async.execute on(#hal.device.optimal<[#hal.device.affinity<@byte>, #hal.device.affinity<@word>]>) with(%input as %captured: !stream.resource<external>{%c8}) -> !stream.resource<external>{%c16} {
    %dispatched = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%captured[%c0 to %c8 for %c8]) : (!stream.resource<external>{%c8}) -> !stream.resource<transient>{%c8}
    %target = stream.async.alloca : !stream.resource<external>{%c16}
    %updated = stream.async.update %dispatched, %target[%c0 to %c8] : !stream.resource<transient>{%c8} -> %target as !stream.resource<external>{%c16}
    %filled = stream.async.fill %c0_i8, %updated[%c8 to %c16 for %c8] : i8 -> %updated as !stream.resource<external>{%c16}
    stream.yield %filled : !stream.resource<external>{%c16}
  } => !stream.timepoint
  %0 = stream.timepoint.await %timepoint => %results : !stream.resource<external>{%c16}
  util.return %0 : !stream.resource<external>
}

// -----

// A set whose device records only a dispatch unit keeps the function's fill unit.

util.global private @device = #hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device

// CHECK-LABEL: @physical_fill_optimal_explicit_config
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.update
//   CHECK-NOT:   stream.async.concurrent
//       CHECK:   stream.async.fill
util.func public @physical_fill_optimal_explicit_config(%input: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">,
                stream.resources = #stream.resource_config<{max_allocation_size = 4096, min_buffer_offset_alignment = 64, max_buffer_range = 2048, min_buffer_range_alignment = 64, index_bits = 32, min_fill_granularity = 4}>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  %c0_i8 = arith.constant 0 : i8
  %results, %timepoint = stream.async.execute on(#hal.device.optimal<[#hal.device.affinity<@device>]>) with(%input as %captured: !stream.resource<external>{%c6}) -> !stream.resource<external>{%c12} {
    %dispatched = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%captured[%c0 to %c6 for %c6]) : (!stream.resource<external>{%c6}) -> !stream.resource<transient>{%c6}
    %target = stream.async.alloca : !stream.resource<external>{%c12}
    %updated = stream.async.update %dispatched, %target[%c0 to %c6] : !stream.resource<transient>{%c6} -> %target as !stream.resource<external>{%c12}
    %filled = stream.async.fill %c0_i8, %updated[%c6 to %c12 for %c6] : i8 -> %updated as !stream.resource<external>{%c12}
    stream.yield %filled : !stream.resource<external>{%c12}
  } => !stream.timepoint
  %0 = stream.timepoint.await %timepoint => %results : !stream.resource<external>{%c12}
  util.return %0 : !stream.resource<external>
}
