// RUN: iree-opt --split-input-file --verify-each --iree-stream-schedule-allocation --verify-diagnostics %s

// Reuse cannot change the resource's lifetime requirements.
util.func public @different_lifetime(%source: !stream.resource<transient>, %size: index) {
  %result, %done = stream.async.execute with(%source as %capture: !stream.resource<transient>{%size}) -> !stream.resource<external>{%size} {
    // expected-error @+1 {{reuse across resource lifetimes is not supported}}
    %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> !stream.resource<external>{%size} storage = reuse
    stream.yield %transfer : !stream.resource<external>{%size}
  } => !stream.timepoint
  util.return
}

// -----

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {unified_memory = true})
  ]>
} {
  // An allocation pinned only to A cannot be assumed accessible from B.
  util.func public @incompatible_allocation(%size: index) {
    %source, %ready = stream.resource.alloca uninitialized on(#hal.device.affinity<@device_a>) : !stream.resource<transient>{%size} => !stream.timepoint
    %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) await(%ready) => with(%source as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
      // expected-error @+1 {{reuse source allocation is not accessible to both devices}}
      %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@device_b>) !stream.resource<transient>{%size} storage = reuse
      stream.yield %transfer : !stream.resource<transient>{%size}
    } => !stream.timepoint
    util.return
  }
}

// -----

// A transfer label cannot move a local producer's execution from A to B.
util.func public @mislabeled_local_source(%size: index, %value: i32) {
  %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with() -> !stream.resource<transient>{%size} {
    %source = stream.async.splat %value : i32 -> !stream.resource<transient>{%size}
    // expected-error @+1 {{reuse local source affinity differs from its producer}}
    %transfer = stream.async.transfer %source : !stream.resource<transient>{%size} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_b>) !stream.resource<transient>{%size} storage = reuse
    stream.yield %transfer : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.return
}

// -----

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {unified_memory = true})
  ]>
} {
  // An imported buffer's actual allocator and usage are not established here.
  util.func public @imported_buffer(%input: !hal.buffer_view, %size: index) {
    %source = stream.tensor.import on(#hal.device.affinity<@device_a>) %input : !hal.buffer_view -> tensor<4xi32> in !stream.resource<external>{%size}
    %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with(%source as %capture: !stream.resource<external>{%size}) -> !stream.resource<external>{%size} {
      // expected-error @+1 {{reuse source allocation is not accessible to both devices}}
      %transfer = stream.async.transfer %capture : !stream.resource<external>{%size} -> to(#hal.device.affinity<@device_b>) !stream.resource<external>{%size} storage = reuse
      stream.yield %transfer : !stream.resource<external>{%size}
    } => !stream.timepoint
    util.return
  }
}

// -----

// Shared topology is required for a cross-device handoff.
util.func public @discrete_devices(%source: !stream.resource<transient>, %size: index) {
  %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with(%source as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    // expected-error @+1 {{reuse requires shared access between source and target devices}}
    %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@device_b>) !stream.resource<transient>{%size} storage = reuse
    stream.yield %transfer : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.return
}

// -----

// A result with a different byte extent cannot share the complete source range.
util.func public @different_size(%source: !stream.resource<transient>, %size: index, %other_size: index) {
  %result, %done = stream.async.execute with(%source as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%other_size} {
    // expected-error @+1 {{reuse requires the source and result to have the same size}}
    %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%other_size} storage = reuse
    stream.yield %transfer : !stream.resource<transient>{%other_size}
  } => !stream.timepoint
  util.return
}

// -----

// An explicit transfer affinity cannot override the provenance of an import.
util.func public @mismatched_import_affinity(%input: !hal.buffer_view, %size: index) {
  %source = stream.tensor.import on(#hal.device.affinity<@device_a>) %input : !hal.buffer_view -> tensor<4xi32> in !stream.resource<external>{%size}
  %result, %done = stream.async.execute on(#hal.device.affinity<@device_b>) with(%source as %capture: !stream.resource<external>{%size}) -> !stream.resource<external>{%size} {
    // expected-error @+1 {{reuse source affinity differs from its imported buffer}}
    %transfer = stream.async.transfer %capture : !stream.resource<external>{%size} from(#hal.device.affinity<@device_b>) -> !stream.resource<external>{%size} storage = reuse
    stream.yield %transfer : !stream.resource<external>{%size}
  } => !stream.timepoint
  util.return
}

// -----

// A local subview needs a reservation for its parent, not a new target copy.
util.func public @local_subview(%size: index, %offset: index, %length: index, %value: i32) {
  %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with() -> !stream.resource<transient>{%length} {
    %source = stream.async.splat %value : i32 -> !stream.resource<transient>{%size}
    %view = stream.resource.subview %source[%offset] : !stream.resource<transient>{%size} -> !stream.resource<transient>{%length}
    // expected-error @+1 {{reuse of a locally produced subview is not supported}}
    %transfer = stream.async.transfer %view : !stream.resource<transient>{%length} -> !stream.resource<transient>{%length} storage = reuse
    stream.yield %transfer : !stream.resource<transient>{%length}
  } => !stream.timepoint
  util.return
}

// -----

// Do not reserve a partial source view before its full reuse result.
util.func public @escaping_local_source_subview(%size: index, %offset: index, %length: index, %value: i32) {
  // expected-error @+1 {{reuse of a locally produced escaping subview is not supported}}
  %results:2, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with() -> (!stream.resource<transient>{%length}, !stream.resource<transient>{%size}) {
    %source = stream.async.splat %value : i32 -> !stream.resource<transient>{%size}
    %view = stream.resource.subview %source[%offset] : !stream.resource<transient>{%size} -> !stream.resource<transient>{%length}
    %transfer = stream.async.transfer %source : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size} storage = reuse
    stream.yield %view, %transfer : !stream.resource<transient>{%length}, !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.return
}

// -----

// The same partial view is unsupported when the full reuse result comes first.
util.func public @escaping_local_source_subview_second(%size: index, %offset: index, %length: index, %value: i32) {
  // expected-error @+1 {{reuse of a locally produced escaping subview is not supported}}
  %results:2, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with() -> (!stream.resource<transient>{%size}, !stream.resource<transient>{%length}) {
    %source = stream.async.splat %value : i32 -> !stream.resource<transient>{%size}
    %view = stream.resource.subview %source[%offset] : !stream.resource<transient>{%size} -> !stream.resource<transient>{%length}
    %transfer = stream.async.transfer %source : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size} storage = reuse
    stream.yield %transfer, %view : !stream.resource<transient>{%size}, !stream.resource<transient>{%length}
  } => !stream.timepoint
  util.return
}

// -----

// Without an execution or default affinity, physical placement is unknown.
util.func public @unresolved_affinity(%source: !stream.resource<transient>, %size: index) {
  %result, %done = stream.async.execute with(%source as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    // expected-error @+1 {{reuse requires resolved source and target affinities}}
    %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size} storage = reuse
    stream.yield %transfer : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.return
}

// -----

// A known allocation without a placement cannot establish device access.
util.func public @unplaced_allocation(%size: index) {
  %source, %ready = stream.resource.alloca uninitialized : !stream.resource<transient>{%size} => !stream.timepoint
  %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) await(%ready) => with(%source as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    // expected-error @+1 {{reuse source allocation has unresolved affinity}}
    %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size} storage = reuse
    stream.yield %transfer : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.return
}

// -----

// An import without a device assertion gives no proof of its allocation.
util.func public @unplaced_import(%input: !hal.buffer_view, %size: index) {
  %source = stream.tensor.import %input : !hal.buffer_view -> tensor<4xi32> in !stream.resource<external>{%size}
  %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with(%source as %capture: !stream.resource<external>{%size}) -> !stream.resource<external>{%size} {
    // expected-error @+1 {{reuse imported buffer has unresolved affinity}}
    %transfer = stream.async.transfer %capture : !stream.resource<external>{%size} -> !stream.resource<external>{%size} storage = reuse
    stream.yield %transfer : !stream.resource<external>{%size}
  } => !stream.timepoint
  util.return
}

// -----

// A transient packed only within the execution region is placed on A, so its
// cross-device result must escape to receive a jointly placed reservation.
module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {unified_memory = true})
  ]>
} {
  util.func public @local_result_does_not_escape(%size: index, %value: i32) {
    %done = stream.async.execute on(#hal.device.affinity<@device_a>) with() {
      %source = stream.async.splat %value : i32 -> !stream.resource<transient>{%size}
      // expected-error @+1 {{cross-device reuse requires a captured shared allocation or a directly escaping local transient producer}}
      %transfer = stream.async.transfer %source : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@device_b>) !stream.resource<transient>{%size} storage = reuse
      stream.yield
    } => !stream.timepoint
    util.return
  }
}
