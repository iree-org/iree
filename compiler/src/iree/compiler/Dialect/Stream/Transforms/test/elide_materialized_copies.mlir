// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-stream-materialize-copy-on-write),iree-stream-elide-async-copies,cse,canonicalize)' %s | FileCheck %s

// This test suite ensures that the key copy-related part of the pipeline
// (materialize + elide copies) produces correct zero-copy code for common
// patterns. MaterializeCopyOnWritePass conservatively inserts clones then
// ElideAsyncCopiesPass removes unnecessary ones using whole-program analysis.
// Each pass is intended to be relatively straightforward at doing what it is
// supposed to but we must ensure that when run together in order we handle
// these key cases correctly.

// Tests KV cache update pattern (common in LLM inference).
// User provides:
//   - step_input: new tokens/embeddings to process
//   - kvcache: existing cache buffer to read from and write to
// Pattern:
//   1. Import both buffers as stream resources
//   2. Slice relevant portion of cache for reading (past KV entries)
//   3. Use cache slice + step input to compute new KV entries
//   4. Update cache with new entries (in-place, no copy needed)
//   5. Alias updated cache back to user's buffer (zero-copy output)
// The cache should NEVER be cloned since:
//   - Read usage (slice) is complete before write (update)
//   - Update writes to different location than read
//   - Final alias just returns the buffer to user

stream.executable private @ex_compute_kv {
  stream.executable.export public @compute_new_kv workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @kvcache_update_step
// CHECK-SAME: (%[[STEP_VIEW:.+]]: !util.buffer, %[[CACHE_VIEW:.+]]: !util.buffer) -> !util.buffer
util.func public @kvcache_update_step(%step_input_view: !util.buffer, %kvcache_view: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index

  // Import step input (new tokens/embeddings to process).
  // CHECK: %[[STEP_SIZE:.+]] = stream.tensor.sizeof
  // CHECK: %[[STEP:.+]] = stream.tensor.import %[[STEP_VIEW]] : !util.buffer -> tensor<1x512xf32> in !stream.resource<external>{%[[STEP_SIZE]]}
  %step_size = stream.tensor.sizeof tensor<1x512xf32> : index
  %step_input = stream.tensor.import %step_input_view : !util.buffer -> tensor<1x512xf32> in !stream.resource<external>{%step_size}

  // Import KV cache (user-provided storage we'll read from and write to).
  // CHECK: %[[CACHE_SIZE:.+]] = stream.tensor.sizeof
  // CHECK: %[[KVCACHE:.+]] = stream.tensor.import %[[CACHE_VIEW]] : !util.buffer -> tensor<100x512xf32> in !stream.resource<external>{%[[CACHE_SIZE]]}
  %cache_size = stream.tensor.sizeof tensor<100x512xf32> : index
  %kvcache = stream.tensor.import %kvcache_view : !util.buffer -> tensor<100x512xf32> in !stream.resource<external>{%cache_size}

  // Slice out relevant cache entries for current step (read-only).
  %slice_size = stream.tensor.sizeof tensor<10x512xf32> : index
  // CHECK: %[[CACHE_SLICE:.+]] = stream.async.slice %[[KVCACHE]]
  %cache_slice = stream.async.slice %kvcache[%c0 to %slice_size] : !stream.resource<external>{%cache_size} -> !stream.resource<external>{%slice_size}

  // Compute new KV entries using step input + cache slice (both read-only).
  // CHECK: %[[NEW_KV:.+]] = stream.async.dispatch @ex_compute_kv::@compute_new_kv[%c1, %c1, %c1](%[[STEP]][%c0 to %[[STEP_SIZE]] for %[[STEP_SIZE]]], %[[CACHE_SLICE]]{{.+}}) : (!stream.resource<external>{%[[STEP_SIZE]]}, !stream.resource<external>{{.+}}) -> !stream.resource<*>{{.+}}
  %new_kv_size = stream.tensor.sizeof tensor<1x512xf32> : index
  %new_kv = stream.async.dispatch @ex_compute_kv::@compute_new_kv[%c1, %c1, %c1](%step_input[%c0 to %step_size for %step_size], %cache_slice[%c0 to %slice_size for %slice_size]) : (!stream.resource<external>{%step_size}, !stream.resource<external>{%slice_size}) -> !stream.resource<*>{%new_kv_size}

  // Update cache with new KV entries at position for current step.
  // This is the critical operation: cache was read earlier (via slice), now we
  // write to it. Since the read is complete and we're writing to a different
  // portion, NO CLONE should be inserted.
  %update_offset = arith.constant 20480 : index  // offset for row 10 (10 * 512 * 4 bytes)
  // CHECK-NOT: stream.async.clone
  // CHECK: %[[UPDATED_CACHE:.+]] = stream.async.update %[[NEW_KV]], %[[KVCACHE]]
  %updated_cache = stream.async.update %new_kv, %kvcache[%update_offset to %new_kv_size] : !stream.resource<*>{%new_kv_size} -> %kvcache as !stream.resource<external>{%cache_size}

  // Alias updated cache back to user's buffer (hal.tensor.alias pattern).
  // This completes the zero-copy pattern: input buffer is mutated and returned.
  // Note: slice and transfer are elided by optimizer (canonicalize folds identity slice, elide-async-copies removes transfer).
  %result_slice = stream.async.slice %updated_cache[%c0 to %cache_size] : !stream.resource<external>{%cache_size} -> !stream.resource<external>{%cache_size}
  %result = stream.async.transfer %result_slice : !stream.resource<external>{%cache_size} -> !stream.resource<external>{%cache_size}

  // Export result back to hal.buffer (completes the alias).
  // CHECK: %[[EXPORTED:.+]] = stream.tensor.export %[[UPDATED_CACHE]]
  %exported = stream.tensor.export %result : tensor<100x512xf32> in !stream.resource<external>{%cache_size} -> !util.buffer

  // Return exported buffer_view.
  // CHECK: util.return %[[EXPORTED]]
  util.return %exported : !util.buffer
}

// -----

// Tests user-provided output storage pattern.
// User provides output buffer, we compute result and write it in-place.

stream.executable private @ex_compute_output {
  stream.executable.export public @compute workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @user_provided_output_storage
// CHECK-SAME: (%[[INPUT_VIEW:.+]]: !util.buffer, %[[OUTPUT_VIEW:.+]]: !util.buffer) -> !util.buffer
util.func public @user_provided_output_storage(%input_view: !util.buffer, %output_view: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Import input.
  // CHECK: %[[INPUT_SIZE:.+]] = stream.tensor.sizeof
  // CHECK: %[[INPUT:.+]] = stream.tensor.import %[[INPUT_VIEW]] : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%[[INPUT_SIZE]]}
  %input_size = stream.tensor.sizeof tensor<1024xf32> : index
  %input = stream.tensor.import %input_view : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%input_size}

  // Import user-provided output storage.
  // CHECK: %[[OUTPUT_STORAGE:.+]] = stream.tensor.import %[[OUTPUT_VIEW]]
  %output_size = stream.tensor.sizeof tensor<1024xf32> : index
  %output_storage = stream.tensor.import %output_view : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%output_size}

  // Compute result from input.
  // CHECK: %[[COMPUTED:.+]] = stream.async.dispatch @ex_compute_output::@compute[%c1, %c1, %c1](%[[INPUT]][%c0 to %[[INPUT_SIZE]] for %[[INPUT_SIZE]]]) : (!stream.resource<external>{%[[INPUT_SIZE]]}) -> !stream.resource<*>{{.+}}
  %computed_size = stream.tensor.sizeof tensor<1024xf32> : index
  %computed = stream.async.dispatch @ex_compute_output::@compute[%c1, %c1, %c1](%input[%c0 to %input_size for %input_size]) : (!stream.resource<external>{%input_size}) -> !stream.resource<*>{%computed_size}

  // Write computed result into user-provided output storage (zero-copy).
  // CHECK-NOT: stream.async.clone
  // CHECK: %[[UPDATED:.+]] = stream.async.update %[[COMPUTED]], %[[OUTPUT_STORAGE]]
  %updated = stream.async.update %computed, %output_storage[%c0 to %computed_size] : !stream.resource<*>{%computed_size} -> %output_storage as !stream.resource<external>{%output_size}

  // Alias back to user (hal.tensor.alias pattern).
  // Note: identity slice and transfer are elided by canonicalize and elide-async-copies.
  %slice = stream.async.slice %updated[%c0 to %computed_size] : !stream.resource<external>{%output_size} -> !stream.resource<external>{%computed_size}
  %transfer = stream.async.transfer %slice : !stream.resource<external>{%computed_size} -> !stream.resource<external>{%computed_size}

  // Export and return (via alias).
  // CHECK: %[[EXPORTED:.+]] = stream.tensor.export %[[UPDATED]]
  %exported = stream.tensor.export %transfer : tensor<1024xf32> in !stream.resource<external>{%computed_size} -> !util.buffer

  // CHECK: util.return %[[EXPORTED]]
  util.return %exported : !util.buffer
}

// -----

// Tests sequence of in-place fills on imported buffer.

// CHECK-LABEL: @inplace_fill_imported_buffer
// CHECK-SAME: (%[[BUFFER_VIEW:.+]]: !util.buffer) -> !util.buffer
util.func public @inplace_fill_imported_buffer(%buffer_view: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32

  // Import user buffer.
  // CHECK: %[[SIZE:.+]] = stream.tensor.sizeof
  // CHECK: %[[BUFFER:.+]] = stream.tensor.import %[[BUFFER_VIEW]] : !util.buffer -> tensor<256xf32> in !stream.resource<external>{%[[SIZE]]}
  %size = stream.tensor.sizeof tensor<256xf32> : index
  %buffer = stream.tensor.import %buffer_view : !util.buffer -> tensor<256xf32> in !stream.resource<external>{%size}

  // Fill first region - block arg gets clone inserted, then elided.
  // CHECK-NOT: stream.async.clone
  // CHECK: %[[FILL1:.+]] = stream.async.fill %c123_i32, %[[BUFFER]][%c0 to %c64 for %c64] : i32 -> %[[BUFFER]] as !stream.resource<external>{%[[SIZE]]}
  %fill1 = stream.async.fill %c123_i32, %buffer[%c0 to %c64 for %c64] : i32 -> %buffer as !stream.resource<external>{%size}

  // Fill second region - in-place.
  // CHECK-NOT: stream.async.clone
  // CHECK: %[[FILL2:.+]] = stream.async.fill %c456_i32, %[[FILL1]][%c64 to %c128 for %c64] : i32 -> %[[FILL1]] as !stream.resource<external>{%[[SIZE]]}
  %fill2 = stream.async.fill %c456_i32, %fill1[%c64 to %c128 for %c64] : i32 -> %fill1 as !stream.resource<external>{%size}

  // Fill third region - in-place.
  // CHECK-NOT: stream.async.clone
  // CHECK: %[[FILL3:.+]] = stream.async.fill %c123_i32, %[[FILL2]][%c128 to %c256 for %c128] : i32 -> %[[FILL2]] as !stream.resource<external>{%[[SIZE]]}
  %fill3 = stream.async.fill %c123_i32, %fill2[%c128 to %c256 for %c128] : i32 -> %fill2 as !stream.resource<external>{%size}

  // Alias back.
  // Note: identity slice and transfer are optimized away by canonicalize and elide-async-copies.
  %slice = stream.async.slice %fill3[%c0 to %size] : !stream.resource<external>{%size} -> !stream.resource<external>{%size}
  %transfer = stream.async.transfer %slice : !stream.resource<external>{%size} -> !stream.resource<external>{%size}
  // CHECK-NOT: stream.async.slice
  // CHECK-NOT: stream.async.transfer
  // CHECK: %[[EXPORTED:.+]] = stream.tensor.export %[[FILL3]] : tensor<256xf32> in !stream.resource<external>{%[[SIZE]]} -> !util.buffer
  %exported = stream.tensor.export %transfer : tensor<256xf32> in !stream.resource<external>{%size} -> !util.buffer

  // CHECK: util.return %[[EXPORTED]]
  util.return %exported : !util.buffer
}

// -----

// Tests SCF loop with in-place mutations on imported buffer.

stream.executable private @ex_mutate {
  stream.executable.export public @mutate workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @scf_for_inplace_on_imported
// CHECK-SAME: (%[[BUFFER_VIEW:.+]]: !util.buffer, %[[COUNT:.+]]: index) -> !util.buffer
util.func public @scf_for_inplace_on_imported(%buffer_view: !util.buffer, %count: index) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Import buffer.
  // CHECK: %[[SIZE:.+]] = stream.tensor.sizeof
  // CHECK: %[[BUFFER:.+]] = stream.tensor.import %[[BUFFER_VIEW]] : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%[[SIZE]]}
  %size = stream.tensor.sizeof tensor<1024xf32> : index
  %buffer = stream.tensor.import %buffer_view : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%size}

  // Loop mutates buffer in-place each iteration.
  // CHECK: %[[RESULT:.+]] = scf.for %{{.+}} = %c0 to %[[COUNT]] step %c1 iter_args(%[[ITER:.+]] = %[[BUFFER]]) -> (!stream.resource<external>) {
  %result = scf.for %i = %c0 to %count step %c1 iter_args(%iter = %buffer) -> (!stream.resource<external>) {
    // CHECK-NOT: stream.async.clone
    // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch @ex_mutate::@mutate[%c1, %c1, %c1](%[[ITER]][%c0 to %[[SIZE]] for %[[SIZE]]]) : (!stream.resource<external>{%[[SIZE]]}) -> %[[ITER]]{%[[SIZE]]}
    %mutated = stream.async.dispatch @ex_mutate::@mutate[%c1, %c1, %c1](%iter[%c0 to %size for %size]) : (!stream.resource<external>{%size}) -> %iter{%size}
    // CHECK: scf.yield %[[DISPATCH]] : !stream.resource<external>
    scf.yield %mutated : !stream.resource<external>
  }

  // Alias back.
  // Note: identity slice and transfer are optimized away.
  %slice = stream.async.slice %result[%c0 to %size] : !stream.resource<external>{%size} -> !stream.resource<external>{%size}
  %transfer = stream.async.transfer %slice : !stream.resource<external>{%size} -> !stream.resource<external>{%size}
  // CHECK-NOT: stream.async.slice
  // CHECK-NOT: stream.async.transfer
  // CHECK: %[[EXPORTED:.+]] = stream.tensor.export %[[RESULT]] : tensor<1024xf32> in !stream.resource<external>{%[[SIZE]]} -> !util.buffer
  %exported = stream.tensor.export %transfer : tensor<1024xf32> in !stream.resource<external>{%size} -> !util.buffer

  // CHECK: util.return %[[EXPORTED]]
  util.return %exported : !util.buffer
}

// -----

// Tests scf.if with in-place mutations in both branches.

// CHECK-LABEL: @scf_if_inplace_both_branches
// CHECK-SAME: (%[[COND:.+]]: i1, %[[BUFFER_VIEW:.+]]: !util.buffer) -> !util.buffer
util.func public @scf_if_inplace_both_branches(%cond: i1, %buffer_view: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32

  // Import buffer.
  // CHECK: %[[SIZE:.+]] = stream.tensor.sizeof
  // CHECK: %[[BUFFER:.+]] = stream.tensor.import %[[BUFFER_VIEW]] : !util.buffer -> tensor<256xf32> in !stream.resource<external>{%[[SIZE]]}
  %size = stream.tensor.sizeof tensor<256xf32> : index
  %buffer = stream.tensor.import %buffer_view : !util.buffer -> tensor<256xf32> in !stream.resource<external>{%size}

  // Conditional in-place mutation.
  // CHECK: %[[RESULT:.+]] = scf.if %[[COND]] -> (!stream.resource<external>) {
  %result = scf.if %cond -> !stream.resource<external> {
    // CHECK-NOT: stream.async.clone
    // CHECK: %[[FILL1:.+]] = stream.async.fill %c123_i32, %[[BUFFER]][%c0 to %c100 for %c100] : i32 -> %[[BUFFER]] as !stream.resource<external>{%[[SIZE]]}
    %fill1 = stream.async.fill %c123_i32, %buffer[%c0 to %c100 for %c100] : i32 -> %buffer as !stream.resource<external>{%size}
    // CHECK: scf.yield %[[FILL1]] : !stream.resource<external>
    scf.yield %fill1 : !stream.resource<external>
  } else {
    // CHECK-NOT: stream.async.clone
    // CHECK: %[[FILL2:.+]] = stream.async.fill %c456_i32, %[[BUFFER]][%c0 to %c100 for %c100] : i32 -> %[[BUFFER]] as !stream.resource<external>{%[[SIZE]]}
    %fill2 = stream.async.fill %c456_i32, %buffer[%c0 to %c100 for %c100] : i32 -> %buffer as !stream.resource<external>{%size}
    // CHECK: scf.yield %[[FILL2]] : !stream.resource<external>
    scf.yield %fill2 : !stream.resource<external>
  }

  // Alias back.
  // Note: identity slice and transfer are optimized away.
  %slice = stream.async.slice %result[%c0 to %size] : !stream.resource<external>{%size} -> !stream.resource<external>{%size}
  %transfer = stream.async.transfer %slice : !stream.resource<external>{%size} -> !stream.resource<external>{%size}
  // CHECK-NOT: stream.async.slice
  // CHECK-NOT: stream.async.transfer
  // CHECK: %[[EXPORTED:.+]] = stream.tensor.export %[[RESULT]] : tensor<256xf32> in !stream.resource<external>{%[[SIZE]]} -> !util.buffer
  %exported = stream.tensor.export %transfer : tensor<256xf32> in !stream.resource<external>{%size} -> !util.buffer

  // CHECK: util.return %[[EXPORTED]]
  util.return %exported : !util.buffer
}

// -----

// Tests dispatch with same buffer used multiple times (aliasing inputs + tied output).

stream.executable private @ex_compute_aliased {
  stream.executable.export public @compute workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @dispatch_aliased_buffer
// CHECK-SAME: (%[[BUFFER_VIEW:.+]]: !util.buffer) -> !util.buffer
util.func public @dispatch_aliased_buffer(%buffer_view: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Import buffer.
  // CHECK: %[[SIZE:.+]] = stream.tensor.sizeof
  // CHECK: %[[BUFFER:.+]] = stream.tensor.import %[[BUFFER_VIEW]] : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%[[SIZE]]}
  %size = stream.tensor.sizeof tensor<1024xf32> : index
  %buffer = stream.tensor.import %buffer_view : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%size}

  // Dispatch uses buffer multiple times (read) and writes to it (tied).
  // CHECK-NOT: stream.async.clone
  // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch @ex_compute_aliased::@compute[%c1, %c1, %c1](%[[BUFFER]][%c0 to %[[SIZE]] for %[[SIZE]]], %[[BUFFER]][%c0 to %[[SIZE]] for %[[SIZE]]]) : (!stream.resource<external>{%[[SIZE]]}, !stream.resource<external>{%[[SIZE]]}) -> %[[BUFFER]]{%[[SIZE]]}
  %result = stream.async.dispatch @ex_compute_aliased::@compute[%c1, %c1, %c1](%buffer[%c0 to %size for %size], %buffer[%c0 to %size for %size]) : (!stream.resource<external>{%size}, !stream.resource<external>{%size}) -> %buffer{%size}

  // Alias back.
  // Note: identity slice and transfer are optimized away.
  %slice = stream.async.slice %result[%c0 to %size] : !stream.resource<external>{%size} -> !stream.resource<external>{%size}
  %transfer = stream.async.transfer %slice : !stream.resource<external>{%size} -> !stream.resource<external>{%size}
  // CHECK-NOT: stream.async.slice
  // CHECK-NOT: stream.async.transfer
  // CHECK: %[[EXPORTED:.+]] = stream.tensor.export %[[DISPATCH]] : tensor<1024xf32> in !stream.resource<external>{%[[SIZE]]} -> !util.buffer
  %exported = stream.tensor.export %transfer : tensor<1024xf32> in !stream.resource<external>{%size} -> !util.buffer

  // CHECK: util.return %[[EXPORTED]]
  util.return %exported : !util.buffer
}

// -----

// Tests nested SCF with in-place mutations.

// CHECK-LABEL: @nested_scf_inplace
// CHECK-SAME: (%[[BUFFER_VIEW:.+]]: !util.buffer) -> !util.buffer
util.func public @nested_scf_inplace(%buffer_view: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32

  // Import buffer.
  // CHECK: %[[SIZE:.+]] = stream.tensor.sizeof
  // CHECK: %[[BUFFER:.+]] = stream.tensor.import %[[BUFFER_VIEW]] : !util.buffer -> tensor<256xf32> in !stream.resource<external>{%[[SIZE]]}
  %size = stream.tensor.sizeof tensor<256xf32> : index
  %buffer = stream.tensor.import %buffer_view : !util.buffer -> tensor<256xf32> in !stream.resource<external>{%size}

  // Nested loops mutating in-place.
  // CHECK: %[[OUTER:.+]] = scf.for %{{.+}} = %c0 to %c5 step %c1 iter_args(%[[OUTER_ITER:.+]] = %[[BUFFER]]) -> (!stream.resource<external>) {
  %result = scf.for %i = %c0 to %c5 step %c1 iter_args(%outer_iter = %buffer) -> (!stream.resource<external>) {
    // CHECK: %[[INNER:.+]] = scf.for %{{.+}} = %c0 to %c5 step %c1 iter_args(%[[INNER_ITER:.+]] = %[[OUTER_ITER]]) -> (!stream.resource<external>) {
    %inner_result = scf.for %j = %c0 to %c5 step %c1 iter_args(%inner_iter = %outer_iter) -> (!stream.resource<external>) {
      // CHECK-NOT: stream.async.clone
      // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[INNER_ITER]][%c0 to %c100 for %c100] : i32 -> %[[INNER_ITER]] as !stream.resource<external>{%[[SIZE]]}
      %fill = stream.async.fill %c123_i32, %inner_iter[%c0 to %c100 for %c100] : i32 -> %inner_iter as !stream.resource<external>{%size}
      // CHECK: scf.yield %[[FILL]] : !stream.resource<external>
      scf.yield %fill : !stream.resource<external>
    }
    // CHECK: scf.yield %[[INNER]] : !stream.resource<external>
    scf.yield %inner_result : !stream.resource<external>
  }

  // Alias back.
  // Note: identity slice and transfer are optimized away.
  %slice = stream.async.slice %result[%c0 to %size] : !stream.resource<external>{%size} -> !stream.resource<external>{%size}
  %transfer = stream.async.transfer %slice : !stream.resource<external>{%size} -> !stream.resource<external>{%size}
  // CHECK-NOT: stream.async.slice
  // CHECK-NOT: stream.async.transfer
  // CHECK: %[[EXPORTED:.+]] = stream.tensor.export %[[OUTER]] : tensor<256xf32> in !stream.resource<external>{%[[SIZE]]} -> !util.buffer
  %exported = stream.tensor.export %transfer : tensor<256xf32> in !stream.resource<external>{%size} -> !util.buffer

  // CHECK: util.return %[[EXPORTED]]
  util.return %exported : !util.buffer
}

// -----

// Tests cross-device operation with unified memory topology.
// When devices have unified memory, transfers should be elided but clones
// may still be needed for copy-on-write semantics.

module @unified_memory attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_a = {unified_memory = true})
  ]>
} {

stream.executable private @ex_cross_device {
  stream.executable.export public @compute workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @cross_device_unified_memory
// CHECK-SAME: (%[[INPUT:.+]]: !util.buffer) -> !util.buffer
util.func public @cross_device_unified_memory(%input: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Import on device A.
  // CHECK: %[[SIZE:.+]] = stream.tensor.sizeof
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import on(#hal.device.promise<@dev_a>) %[[INPUT]] : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%[[SIZE]]}
  %size = stream.tensor.sizeof tensor<1024xf32> : index
  %imported = stream.tensor.import on(#hal.device.promise<@dev_a>) %input : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%size}

  // Dispatch on device B consumes resource from device A.
  // Since devices have unified memory, no explicit transfer should remain.
  // CHECK-NOT: stream.async.transfer
  // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch on(#hal.device.promise<@dev_b>) @ex_cross_device::@compute[%c1, %c1, %c1](%[[IMPORTED]][%c0 to %[[SIZE]] for %[[SIZE]]]) : (!stream.resource<external>{%[[SIZE]]}) -> !stream.resource<*>{%[[SIZE]]}
  %result = stream.async.dispatch on(#hal.device.promise<@dev_b>) @ex_cross_device::@compute[%c1, %c1, %c1](%imported[%c0 to %size for %size]) : (!stream.resource<external>{%size}) -> !stream.resource<*>{%size}

  // Export result.
  // CHECK: %[[EXPORTED:.+]] = stream.tensor.export %[[DISPATCH]] : tensor<1024xf32> in !stream.resource<*>{%[[SIZE]]} -> !util.buffer
  %exported = stream.tensor.export %result : tensor<1024xf32> in !stream.resource<*>{%size} -> !util.buffer

  // CHECK: util.return %[[EXPORTED]]
  util.return %exported : !util.buffer
}

} // module

// -----

// Tests cross-device operation WITHOUT unified memory.
// Transfers between non-unified devices must be preserved.

module @non_unified_memory attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_c = {})  // No unified memory.
  ]>
} {

stream.executable private @ex_cross_device_non_unified {
  stream.executable.export public @compute workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @cross_device_non_unified_memory
// CHECK-SAME: (%[[INPUT:.+]]: !util.buffer) -> !util.buffer
util.func public @cross_device_non_unified_memory(%input: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Import on device A.
  // CHECK: %[[SIZE:.+]] = stream.tensor.sizeof
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import on(#hal.device.promise<@dev_a>) %[[INPUT]] : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%[[SIZE]]}
  %size = stream.tensor.sizeof tensor<1024xf32> : index
  %imported = stream.tensor.import on(#hal.device.promise<@dev_a>) %input : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%size}

  // Cross-device transfer (inserted by earlier passes like RefineUsage).
  // Since devices do NOT have unified memory, transfer MUST be preserved.
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[IMPORTED]] : !stream.resource<external>{%[[SIZE]]} from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_c>) !stream.resource<external>{%[[SIZE]]}
  %transferred = stream.async.transfer %imported : !stream.resource<external>{%size}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_c>) !stream.resource<external>{%size}

  // Dispatch on device C consumes transferred resource.
  // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch on(#hal.device.promise<@dev_c>) @ex_cross_device_non_unified::@compute[%c1, %c1, %c1](%[[TRANSFER]][%c0 to %[[SIZE]] for %[[SIZE]]]) : (!stream.resource<external>{%[[SIZE]]}) -> !stream.resource<*>{%[[SIZE]]}
  %result = stream.async.dispatch on(#hal.device.promise<@dev_c>) @ex_cross_device_non_unified::@compute[%c1, %c1, %c1](%transferred[%c0 to %size for %size]) : (!stream.resource<external>{%size}) -> !stream.resource<*>{%size}

  // Export result.
  // CHECK: %[[EXPORTED:.+]] = stream.tensor.export %[[DISPATCH]] : tensor<1024xf32> in !stream.resource<*>{%[[SIZE]]} -> !util.buffer
  %exported = stream.tensor.export %result : tensor<1024xf32> in !stream.resource<*>{%size} -> !util.buffer

  // CHECK: util.return %[[EXPORTED]]
  util.return %exported : !util.buffer
}

} // module

// -----

// Tests multi-device fork: resource used by operations on different devices.
// With unified memory, no transfers needed but clones may be required for
// copy-on-write if operations mutate the resource.

module @multi_device_fork attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_a = {unified_memory = true})
  ]>
} {

stream.executable private @ex_fork {
  stream.executable.export public @mutate workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @multi_device_fork_unified
// CHECK-SAME: (%[[INPUT:.+]]: !util.buffer) -> (!util.buffer, !util.buffer)
util.func public @multi_device_fork_unified(%input: !util.buffer) -> (!util.buffer, !util.buffer) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Import on device A.
  // CHECK: %[[SIZE:.+]] = stream.tensor.sizeof
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import on(#hal.device.promise<@dev_a>) %[[INPUT]] : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%[[SIZE]]}
  %size = stream.tensor.sizeof tensor<1024xf32> : index
  %imported = stream.tensor.import on(#hal.device.promise<@dev_a>) %input : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%size}

  // Mutations on both devices require clone to avoid conflict.
  // CHECK: %[[CLONE:.+]] = stream.async.clone on(#hal.device.promise<@dev_a>) %[[IMPORTED]] : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  // CHECK-NOT: stream.async.transfer
  // First mutation on device A uses the clone.
  // CHECK: %[[DISPATCH_A:.+]] = stream.async.dispatch on(#hal.device.promise<@dev_a>) @ex_fork::@mutate[%c1, %c1, %c1](%[[CLONE]][%c0 to %[[SIZE]] for %[[SIZE]]]) : (!stream.resource<external>{%[[SIZE]]}) -> %[[CLONE]]{%[[SIZE]]}
  %result_a = stream.async.dispatch on(#hal.device.promise<@dev_a>) @ex_fork::@mutate[%c1, %c1, %c1](%imported[%c0 to %size for %size]) : (!stream.resource<external>{%size}) -> %imported{%size}

  // Second mutation on device B uses the original.
  // CHECK: %[[DISPATCH_B:.+]] = stream.async.dispatch on(#hal.device.promise<@dev_b>) @ex_fork::@mutate[%c1, %c1, %c1](%[[IMPORTED]][%c0 to %[[SIZE]] for %[[SIZE]]]) : (!stream.resource<external>{%[[SIZE]]}) -> %[[IMPORTED]]{%[[SIZE]]}
  %result_b = stream.async.dispatch on(#hal.device.promise<@dev_b>) @ex_fork::@mutate[%c1, %c1, %c1](%imported[%c0 to %size for %size]) : (!stream.resource<external>{%size}) -> %imported{%size}

  // Export both results.
  // CHECK: %[[EXPORT_A:.+]] = stream.tensor.export %[[DISPATCH_A]] : tensor<1024xf32> in !stream.resource<external>{%[[SIZE]]} -> !util.buffer
  // CHECK: %[[EXPORT_B:.+]] = stream.tensor.export %[[DISPATCH_B]] : tensor<1024xf32> in !stream.resource<external>{%[[SIZE]]} -> !util.buffer
  %export_a = stream.tensor.export %result_a : tensor<1024xf32> in !stream.resource<external>{%size} -> !util.buffer
  %export_b = stream.tensor.export %result_b : tensor<1024xf32> in !stream.resource<external>{%size} -> !util.buffer

  // CHECK: util.return %[[EXPORT_A]], %[[EXPORT_B]]
  util.return %export_a, %export_b : !util.buffer, !util.buffer
}

} // module

// -----

// Tests transfer chain collapsing: A->B->C where B has unified memory with both.
// The chain should collapse to direct communication between A and C.

module @transfer_chain attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_a = {unified_memory = true}),
    (@dev_b -> @dev_c = {unified_memory = true}),
    (@dev_c -> @dev_b = {unified_memory = true}),
    (@dev_a -> @dev_c = {})  // No direct unified memory.
  ]>
} {

// CHECK-LABEL: @transfer_chain_collapse
// CHECK-SAME: (%[[INPUT:.+]]: !util.buffer) -> !util.buffer
util.func public @transfer_chain_collapse(%input: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index

  // Import on device A.
  // CHECK: %[[SIZE:.+]] = stream.tensor.sizeof
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import on(#hal.device.promise<@dev_a>) %[[INPUT]] : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%[[SIZE]]}
  %size = stream.tensor.sizeof tensor<1024xf32> : index
  %imported = stream.tensor.import on(#hal.device.promise<@dev_a>) %input : !util.buffer -> tensor<1024xf32> in !stream.resource<external>{%size}

  // Transfer A->B (unified memory, should be elided).
  // Transfer B->C (unified memory, should be elided).
  // Result: no intermediate transfers, direct use of imported resource.
  %transfer_to_b = stream.async.transfer %imported : !stream.resource<external>{%size}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<external>{%size}
  %transfer_to_c = stream.async.transfer %transfer_to_b : !stream.resource<external>{%size}
      from(#hal.device.promise<@dev_b>) -> to(#hal.device.promise<@dev_c>) !stream.resource<external>{%size}

  // Export on device C.
  // CHECK: %[[EXPORTED:.+]] = stream.tensor.export %[[IMPORTED]] : tensor<1024xf32> in !stream.resource<external>{%[[SIZE]]} -> !util.buffer
  %exported = stream.tensor.export %transfer_to_c : tensor<1024xf32> in !stream.resource<external>{%size} -> !util.buffer

  // CHECK: util.return %[[EXPORTED]]
  util.return %exported : !util.buffer
}

} // module
