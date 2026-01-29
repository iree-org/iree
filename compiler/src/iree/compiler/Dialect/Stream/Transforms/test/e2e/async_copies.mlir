// RUN: iree-opt --split-input-file --iree-stream-elide-async-copies --iree-stream-refine-usage --iree-stream-elide-async-copies %s | FileCheck %s

// Tests the full pipeline interaction for cross-lifetime clone elision:
//   1. First ElideAsyncCopies: preserves cross-lifetime clones (types differ)
//   2. RefineUsage: propagates source usage through immutable clones, unifying
//      types on both sides of the clone
//   3. Second ElideAsyncCopies: elides the now-same-type clone

// An immutable clone of an external resource (no tied uses on the result)
// should be fully elided after the pipeline unifies the types. The source
// only has one use (the clone), so it's trivially safe to elide.

stream.executable private @ex {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%in: !stream.binding, %out: !stream.binding) {
      return
    }
  }
}

// CHECK-LABEL: @immutable_clone_single_use
util.func private @immutable_clone_single_use(%size: index, %buf: !hal.buffer_view) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %external = stream.tensor.import on(#hal.device.affinity<@device>)
      %buf : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%size}
  // Clone result has no tied uses. RefineUsage unifies types to external.
  // Second ElideAsyncCopies elides the clone (source has one use).
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone on(#hal.device.affinity<@device>)
      %external : !stream.resource<external>{%size} -> !stream.resource<*>{%size}
  // Dispatch reads directly from the imported resource after clone elision.
  // CHECK: stream.async.dispatch
  %result = stream.async.dispatch on(#hal.device.affinity<@device>)
      @ex::@dispatch(%clone[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  util.return %result : !stream.resource<*>
}

// -----

// A mutated clone of an external resource (dispatch writes in-place via tied
// result) should NOT be elided: the clone provides data isolation so the
// mutation doesn't corrupt the original resource.

stream.executable private @ex2 {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%binding: !stream.binding) {
      return
    }
  }
}

// CHECK-LABEL: @mutated_clone_preserved
util.func private @mutated_clone_preserved(%size: index, %buf: !hal.buffer_view) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %external = stream.tensor.import on(#hal.device.affinity<@device>)
      %buf : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%size}
  // Clone result is mutated in-place by dispatch (tied). ResourceUsageAnalysis
  // does NOT propagate source usage, so the clone result gets its own lifetime.
  // CHECK: stream.async.clone
  %clone = stream.async.clone on(#hal.device.affinity<@device>)
      %external : !stream.resource<external>{%size} -> !stream.resource<*>{%size}
  // CHECK: stream.async.dispatch
  %result = stream.async.dispatch on(#hal.device.affinity<@device>)
      @ex2::@dispatch(%clone[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> %clone{%size}
  util.return %result : !stream.resource<*>
}

// -----

// An immutable clone where the source is also mutated elsewhere. Even after
// RefineUsage unifies the types, the clone cannot be elided because removing
// it would create a write-after-read hazard between the read dispatch and
// the tied (mutating) dispatch.

stream.executable private @ex3 {
  stream.executable.export public @dispatch_read
  stream.executable.export public @dispatch_mutate
  builtin.module {
    func.func @dispatch_read(%in: !stream.binding, %out: !stream.binding) {
      return
    }
    func.func @dispatch_mutate(%binding: !stream.binding) {
      return
    }
  }
}

// CHECK-LABEL: @immutable_clone_source_mutated_elsewhere
util.func private @immutable_clone_source_mutated_elsewhere(%size: index, %buf: !hal.buffer_view) -> (!stream.resource<*>, !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %external = stream.tensor.import on(#hal.device.affinity<@device>)
      %buf : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%size}
  // Clone result has no tied uses, but source is mutated by a subsequent tied
  // dispatch. The clone is needed for data isolation.
  // CHECK: stream.async.clone
  %clone = stream.async.clone on(#hal.device.affinity<@device>)
      %external : !stream.resource<external>{%size} -> !stream.resource<*>{%size}
  // Read from clone (safe, independent copy).
  %read = stream.async.dispatch on(#hal.device.affinity<@device>)
      @ex3::@dispatch_read(%clone[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // Mutate source in-place (tied result). This is why the clone is needed.
  %mutated = stream.async.dispatch on(#hal.device.affinity<@device>)
      @ex3::@dispatch_mutate(%external[%c0 to %size for %size])
      : (!stream.resource<external>{%size}) -> %external{%size}
  util.return %read, %mutated : !stream.resource<*>, !stream.resource<external>
}
