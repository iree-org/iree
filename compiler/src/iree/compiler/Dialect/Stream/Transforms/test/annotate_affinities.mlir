// RUN: iree-opt --split-input-file --iree-stream-annotate-affinities --iree-stream-affinity-solver-max-iterations=8 %s | FileCheck %s

// Note: nothing in here is crazy enough that it should trigger the max
// iteration count. Nearly everything should complete in 1-4 iterations.

// Tests that we can track affinity through optimization barriers. They're meant
// to block optimization but we really can't do much if we don't track affinity.
// We could change this in the future but tests would be harder to write and
// there's not a lot that can be done with an unassigned resource.

// CHECK-LABEL: @optimization_barrier_consumer
util.func private @optimization_barrier_consumer() -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: util.optimization_barrier
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %cst_dno = util.optimization_barrier %cst : tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.transfer %cst_dno : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %cst_a : tensor<1xi32>
}

// -----

// CHECK-LABEL: @optimization_barrier_producer
util.func private @optimization_barrier_producer() -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: util.optimization_barrier
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a_dno = util.optimization_barrier %cst_a : tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %cst_a_dno : tensor<1xi32>
}

// -----

// Tests that constant-like ops get placed with their consumer(s).
// We want to replicate constants where they are consumed instead of performing
// transfers at runtime to move them around and by placing with consumers we
// can know when we need to do that early on.

// CHECK-LABEL: @constant_op
util.func private @constant_op() -> (tensor<1xi32>, tensor<1xi32>) {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.transfer %cst : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %cst_b = flow.tensor.transfer %cst : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_b>]]
  util.return %cst_a, %cst_b : tensor<1xi32>, tensor<1xi32>
}

// -----

// Tests that splats (not constant-like but no consumed values) are placed with
// their consumer(s). These are always best to rematerialize where they are
// consumed to avoid allocating/transferring a bunch of repeated values.

// CHECK-LABEL: @splat_op
util.func private @splat_op() -> tensor<1xi32> {
  %splat_value = arith.constant 123 : i32
  // CHECK: flow.tensor.splat
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %splat = flow.tensor.splat %splat_value : tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %splat_a = flow.tensor.transfer %splat : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %splat_a : tensor<1xi32>
}

// -----

// Tests that splats which cannot be resolved to a single consumer get tagged
// with both.

// CHECK-LABEL: @splat_op_ambiguous
util.func private @splat_op_ambiguous() -> (tensor<1xi32>, tensor<1xi32>) {
  %splat_value = arith.constant 123 : i32
  // CHECK: flow.tensor.splat
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %splat = flow.tensor.splat %splat_value : tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %splat_a = flow.tensor.transfer %splat : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_b>]]
  %splat_b = flow.tensor.transfer %splat : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_b>]]
  util.return %splat_a, %splat_b : tensor<1xi32>, tensor<1xi32>
}

// -----

// Tests that imported tensor placement is inherited.
// Frontends can use this to declare where they expect their arguments to
// be living at the time the functions are invoked. Imports do not perform
// transfers so we must use whatever is declared.

// CHECK-LABEL: @imported_hal_tensor
util.func public @imported_hal_tensor(%buffer_view: !hal.buffer_view, %fence: !hal.fence) -> tensor<1xi32> {
  // CHECK: hal.tensor.import
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %tensor = hal.tensor.import on(#hal.device.promise<@dev_a>) wait(%fence) => %buffer_view "input" : !hal.buffer_view -> tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %tensor : tensor<1xi32>
}

// -----

// CHECK-LABEL: @imported_stream_tensor
util.func public @imported_stream_tensor(%buffer_view: !hal.buffer_view) -> !stream.resource<external> {
  %size = stream.tensor.sizeof on(#hal.device.promise<@dev_a>) tensor<1xi32> : index
  // CHECK: stream.tensor.import
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %resource = stream.tensor.import on(#hal.device.promise<@dev_a>) %buffer_view : !hal.buffer_view -> tensor<1xi32> in !stream.resource<external>{%size}
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %resource : !stream.resource<external>
}

// -----

// Tests that consumer-placed ops exported to buffers are properly placed.
// Frontends can use this to explicitly define where exported tensors must live.
// With consumer-placed ops like constants or splats we place them directly on
// the export target.

// CHECK-LABEL: @exported_hal_constant
util.func public @exported_hal_constant(%fence: !hal.fence) -> !hal.buffer_view {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: hal.tensor.barrier
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_ready = hal.tensor.barrier join(%cst : tensor<1xi32>) => %fence : !hal.fence
  // CHECK: hal.tensor.export
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  %buffer_view = hal.tensor.export on(#hal.device.promise<@dev_a>) %cst_ready "output" : tensor<1xi32> -> !hal.buffer_view
  util.return %buffer_view : !hal.buffer_view
}

// -----

// CHECK-LABEL: @exported_stream_constant
util.func public @exported_stream_constant() -> !hal.buffer_view {
  %size = stream.tensor.sizeof on(#hal.device.promise<@dev_a>) tensor<1x5x64xi32> : index
  // CHECK: stream.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst = stream.tensor.constant : tensor<1x5x64xi32> in !stream.resource<external> = dense<0> : tensor<1x5x64xi32>
  // CHECK: stream.tensor.export
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  %buffer_view = stream.tensor.export on(#hal.device.promise<@dev_a>) %cst : tensor<1x5x64xi32> in !stream.resource<external>{%size} -> !hal.buffer_view
  util.return %buffer_view : !hal.buffer_view
}

// -----

// Tests that producer-placed ops exported to buffers get the appropriate
// affinity on both devices. Frontends can use this to explicitly define where
// exported tensors must live. Transfers may need to be inserted in order to
// respect the required affinities. Note here that the operand to the export
// is on @dev_a instead of the requested @dev_b.

// CHECK-LABEL: @exported_producer
util.func public @exported_producer(%fence: !hal.fence) -> !hal.buffer_view {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.transfer %cst : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: flow.tensor.clone
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %clone_a = flow.tensor.clone %cst_a : tensor<1xi32>
  // CHECK: hal.tensor.barrier
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]}
  %clone_ready_a = hal.tensor.barrier join(%clone_a : tensor<1xi32>) => %fence : !hal.fence
  // CHECK: hal.tensor.export
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %buffer_view = hal.tensor.export on(#hal.device.promise<@dev_b>) %clone_ready_a "output" : tensor<1xi32> -> !hal.buffer_view
  // CHECK: util.return
  util.return %buffer_view : !hal.buffer_view
}

// -----

// Test in-place aliased storage for results.
// Frontends require that the storage be placed as indicated even if that means
// introducing transfers such that the operation is not in-place.

// CHECK-LABEL: @aliased_storage
util.func public @aliased_storage(%view: !hal.buffer_view, %storage: !hal.buffer, %fence: !hal.fence) -> tensor<4xi32> {
  // CHECK: hal.tensor.import
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %arg_a = hal.tensor.import on(#hal.device.promise<@dev_a>) %view : !hal.buffer_view -> tensor<4xi32>
  // CHECK: flow.dispatch @dispatch
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %ret_b = flow.dispatch @dispatch(%arg_a) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: hal.tensor.alias
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %alias_b = hal.tensor.alias on(#hal.device.promise<@dev_b>) %ret_b : tensor<4xi32> to %storage : !hal.buffer
  // CHECK: hal.tensor.barrier
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %barrier = hal.tensor.barrier join(%alias_b : tensor<4xi32>) => %fence : !hal.fence
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  util.return %barrier : tensor<4xi32>
}

// -----

// Tests aliased storage through tied dispatches.

// CHECK-LABEL: @tied_aliased_storage
util.func public @tied_aliased_storage(%view: !hal.buffer_view, %storage: !hal.buffer, %fence: !hal.fence) {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst = flow.tensor.constant dense<123> : tensor<4xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.transfer %cst : tensor<4xi32> to #hal.device.promise<@dev_a>
  // CHECK: flow.dispatch @dispatch0
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %t0 = flow.dispatch @dispatch0(%cst_a) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: flow.dispatch @dispatch1
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %t1 = flow.dispatch @dispatch1(%t0) : (tensor<4xi32>) -> %t0
  // CHECK: hal.tensor.alias
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]}
  %alias = hal.tensor.alias on(#hal.device.promise<@dev_b>) %t1 : tensor<4xi32> to %storage : !hal.buffer
  // CHECK: hal.tensor.barrier
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  hal.tensor.barrier join(%alias : tensor<4xi32>) => %fence : !hal.fence
  util.return
}

// -----

// Tests that consumer-placed ops that pass through tied ops get attributed to
// a single consumer.

// CHECK-LABEL: @tied_constant
util.func private @tied_constant() -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: flow.dispatch @a
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %tied = flow.dispatch @a(%cst) : (tensor<1xi32>) -> %cst
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %tied_a = flow.tensor.transfer %tied : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %tied_a : tensor<1xi32>
}

// -----

// Tests that consumer-placed ops that pass through tied ops get attributed to
// transitive consumers. This is not ideal but allows the application of
// replication policies.

// CHECK-LABEL: @tied_constant_multi_consumer
util.func private @tied_constant_multi_consumer() -> (tensor<1xi32>, tensor<1xi32>) {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: flow.dispatch @a
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %tied_0 = flow.dispatch @a(%cst) : (tensor<1xi32>) -> %cst
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %tied_0_a = flow.tensor.transfer %tied_0 : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: flow.dispatch @b
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %tied_1 = flow.dispatch @b(%cst) : (tensor<1xi32>) -> %cst
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_b>]]
  %tied_1_b = flow.tensor.transfer %tied_1 : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_b>]]
  util.return %tied_0_a, %tied_1_b : tensor<1xi32>, tensor<1xi32>
}

// -----

// Tests the proper transfer of consumer-placed values prior to multiple tied
// uses don't pollute the execution affinity of ops after transfers. Note that
// the constant will still have multiple affinities to allow for policies that
// replicate the constant.

// CHECK-LABEL: @tied_transfer_constant_multi_consumer
util.func private @tied_transfer_constant_multi_consumer() -> (tensor<1xi32>, tensor<1xi32>) {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.transfer %cst : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: flow.dispatch @a
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %tied_0 = flow.dispatch @a(%cst_a) : (tensor<1xi32>) -> %cst_a
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %tied_0_a = flow.tensor.transfer %tied_0 : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %cst_b = flow.tensor.transfer %cst : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: flow.dispatch @b
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %tied_1 = flow.dispatch @b(%cst_b) : (tensor<1xi32>) -> %cst_b
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %tied_1_b = flow.tensor.transfer %tied_1 : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_b>]]
  util.return %tied_0_a, %tied_1_b : tensor<1xi32>, tensor<1xi32>
}

// -----

// Tests that implicitly placed consumers use their transfer execution affinity.

// CHECK-LABEL: @transfer_execution_affinity
util.func private @transfer_execution_affinity() -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %cst_b = flow.tensor.transfer %cst_a : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: flow.dispatch @dispatch
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %dispatch_b = flow.dispatch @dispatch(%cst_b) : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  util.return %dispatch_b : tensor<1xi32>
}

// -----

// Tests that explicitly placed consumers use their explicit execution affinity.

// CHECK-LABEL: @explicit_execution_affinity
util.func private @explicit_execution_affinity() -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: flow.dispatch @dispatch
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %dispatch_b = flow.dispatch @dispatch(%cst_a) {stream.affinity = #hal.device.promise<@dev_b>} : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  util.return %dispatch_b : tensor<1xi32>
}

// -----

// Tests that consumers of operands with multiple affinities inherit those
// affinities for execution. This allows policies to determine where they want
// to execute out of the resources they may be consuming.

// CHECK-LABEL: @consume_multi_affinities
util.func private @consume_multi_affinities() -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %cst_b = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_b>} dense<456> : tensor<1xi32>
  // CHECK: flow.dispatch @dispatch
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %dispatch_ab = flow.dispatch @dispatch(%cst_a, %cst_b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  util.return %dispatch_ab : tensor<1xi32>
}

// -----

// Tests that globals are placed where they are loaded.

// CHECK: util.global private @consumed_global_a
// CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
util.global private @consumed_global_a : tensor<1xi32>
util.func private @consumer_fn() -> tensor<1xi32> {
  // CHECK: util.global.load @consumed_global_a
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %load = util.global.load @consumed_global_a : tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %load_a = flow.tensor.transfer %load : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %load_a : tensor<1xi32>
}

// -----

// Tests that a global loaded from two locations is attributed to both
// affinities. This allows policies to decide whether to replicate the global.

// CHECK: util.global private @consumed_global_ab
// CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]
util.global private @consumed_global_ab : tensor<1xi32>
util.func private @consumer_fn_a() -> tensor<1xi32> {
  // CHECK: util.global.load @consumed_global_ab
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %load = util.global.load @consumed_global_ab : tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %load_a = flow.tensor.transfer %load : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %load_a : tensor<1xi32>
}
util.func private @consumer_fn_b() -> tensor<1xi32> {
  // CHECK: util.global.load @consumed_global_ab
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %load = util.global.load @consumed_global_ab : tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %load_b = flow.tensor.transfer %load : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  util.return %load_b : tensor<1xi32>
}

// -----

// Tests that consumer-placed ops track through global loads.

// CHECK: util.global private mutable @global_b
// CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
util.global private mutable @global_b : tensor<1xi32>
util.func private @producer_fn() {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: util.global.store
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.global.store %cst_a, @global_b : tensor<1xi32>
  util.return
}
util.func private @consumer_fn() -> tensor<1xi32> {
  // CHECK: util.global.load
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %load = util.global.load @global_b : tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %load_b = flow.tensor.transfer %load : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  util.return %load_b : tensor<1xi32>
}

// -----

// Tests that globals that are only stored take the fallback placement of
// their producer. This is silly but can arise prior to global optimization
// passes that may elide them.

// CHECK: util.global private mutable @global_a
// CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
util.global private mutable @global_a : tensor<1xi32>
util.func private @producer_fn() {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: util.global.store
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.global.store %cst_a, @global_a : tensor<1xi32>
  util.return
}

// -----

// Tests that global consumers that take on consumed affinity track the global.

// CHECK: util.global private @global_a
// CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
util.global private @global_a {stream.affinity = #hal.device.promise<@dev_a>} : tensor<1xi32>
// CHECK: util.global private @global_b
// CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
util.global private @global_b {stream.affinity = #hal.device.promise<@dev_b>} : tensor<1xi32>
util.func private @consumer_fn() -> tensor<1xi32> {
  // CHECK: util.global.load @global_a
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %load_a = util.global.load @global_a : tensor<1xi32>
  // CHECK: util.global.load @global_b
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %load_b = util.global.load @global_b : tensor<1xi32>
  // CHECK: flow.dispatch @dispatch
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %result_ab = flow.dispatch @dispatch(%load_a, %load_b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  util.return %result_ab : tensor<1xi32>
}

// -----

// Tests a global update tick that operates on the global from multiple
// affinities.

// CHECK: util.global private mutable @global_a
// CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
util.global private mutable @global_a {stream.affinity = #hal.device.promise<@dev_a>} = dense<123> : tensor<1xi32>
util.func private @step(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: util.global.load @global_a
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %load_a = util.global.load @global_a : tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %arg0_b = flow.tensor.transfer %arg0 : tensor<2xi32> to #hal.device.promise<@dev_b>
  // CHECK: flow.dispatch @dispatch
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>], [#hal.device.promise<@dev_b>]]
  %result_b:2 = flow.dispatch @dispatch(%load_a, %arg0_b) {stream.affinity = #hal.device.promise<@dev_b>} : (tensor<1xi32>, tensor<2xi32>) -> (tensor<1xi32>, tensor<2xi32>)
  // CHECK: util.global.store
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  util.global.store %result_b#0, @global_a : tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  util.return %result_b#1 : tensor<2xi32>
}

// -----

// Tests that constants passed through selects are placed on the consumer.

// CHECK-LABEL: @select_constants_consumed
util.func private @select_constants_consumed(%cond: i1) -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_123 = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_456 = flow.tensor.constant dense<456> : tensor<1xi32>
  // CHECK: arith.select
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst = arith.select %cond, %cst_123, %cst_456 : tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.transfer %cst : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %cst_a : tensor<1xi32>
}

// -----

// Tests that placed operands passed through selects are tracked on consumers.

// CHECK-LABEL: @select_constants_placed
util.func private @select_constants_placed(%cond: i1) -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %cst_b = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_b>} dense<456> : tensor<1xi32>
  // CHECK: arith.select
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %cst_ab = arith.select %cond, %cst_a, %cst_b : tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  util.return %cst_ab : tensor<1xi32>
}

// -----

// Tests that a callee that does not touch an argument still tracks the
// affinity through it.

// CHECK-LABEL: @passthrough_caller
util.func private @passthrough_caller() -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: util.call @passthrough_callee
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %result_a = util.call @passthrough_callee(%cst_a) : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %result_a : tensor<1xi32>
}
// CHECK: util.func private @passthrough_callee
util.func private @passthrough_callee(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %arg0 : tensor<1xi32>
}

// -----

// Tests that callees that consumer-placed arguments that are passed to callees
// get placed based on callee usage.

// CHECK-LABEL: @consumer_placement_caller
util.func private @consumer_placement_caller() -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: util.call @consumer_placement_callee
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %result_a = util.call @consumer_placement_callee(%cst) : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %result_a : tensor<1xi32>
}
// CHECK: util.func private @consumer_placement_callee
util.func private @consumer_placement_callee(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %arg0_a = flow.tensor.transfer %arg0 : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %arg0_a : tensor<1xi32>
}

// -----

// Tests that multiple potential affinities are propagated across call edges.

// CHECK-LABEL: @select_caller
util.func private @select_caller(%arg0: tensor<1xi32>, %cond: i1) -> tensor<1xi32> {
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %arg0_a = flow.tensor.transfer %arg0 : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.call @select_callee
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %result_ab = util.call @select_callee(%arg0_a, %cond) : (tensor<1xi32>, i1) -> tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  util.return %result_ab : tensor<1xi32>
}
// CHECK: util.func private @select_callee
util.func private @select_callee(%arg0_a: tensor<1xi32>, %cond: i1) -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %cst_b = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_b>} dense<123> : tensor<1xi32>
  // CHECK: arith.select
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %select_ab = arith.select %cond, %arg0_a, %cst_b : tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  util.return %select_ab : tensor<1xi32>
}

// -----

// Tests that consumer-placed ops are propagated across call edges.

// CHECK-LABEL: @consumer_multi_placement_caller
util.func private @consumer_multi_placement_caller() -> (tensor<1xi32>, tensor<1xi32>) {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_c>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_c>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: util.call @consumer_multi_placement_callee
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_c>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_c>]]
  %result_0_c = util.call @consumer_multi_placement_callee(%cst) : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_c>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %result_0_a = flow.tensor.transfer %result_0_c : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.call @consumer_multi_placement_callee
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_c>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_c>]]
  %result_1_c = util.call @consumer_multi_placement_callee(%cst) : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_c>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %result_1_b = flow.tensor.transfer %result_1_c : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_b>]]
  util.return %result_0_a, %result_1_b : tensor<1xi32>, tensor<1xi32>
}
// CHECK: util.func private @consumer_multi_placement_callee
util.func private @consumer_multi_placement_callee(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_c>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_c>]]
  %arg0_c = flow.tensor.transfer %arg0 : tensor<1xi32> to #hal.device.promise<@dev_c>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_c>]]
  util.return %arg0_c : tensor<1xi32>
}

// -----

// Tests that operand/result affinities are tracked across call edges.

// CHECK-LABEL: @dispatch_fn_a
util.func private @dispatch_fn_a() -> tensor<4xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %0 = flow.tensor.constant dense<123> : tensor<4xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %1 = flow.tensor.transfer %0 : tensor<4xi32> to #hal.device.promise<@dev_a>
  // CHECK: flow.dispatch @dispatch_a_0
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %2 = flow.dispatch @dispatch_a_0(%1) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: util.call @dispatch_fn_b
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %3 = util.call @dispatch_fn_b(%2) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %4 = flow.tensor.transfer %3 : tensor<4xi32> to #hal.device.promise<@dev_a>
  // CHECK: flow.dispatch @dispatch_a_1
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %5 = flow.dispatch @dispatch_a_1(%4) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %5 : tensor<4xi32>
}
// CHECK: util.func private @dispatch_fn_b
util.func private @dispatch_fn_b(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %0 = flow.tensor.transfer %arg0 : tensor<4xi32> to #hal.device.promise<@dev_b>
  // CHECK: flow.dispatch @dispatch_b
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %1 = flow.dispatch @dispatch_b(%0) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  util.return %1 : tensor<4xi32>
}

// -----

// Tests a realistic call graph with explicit transfers.

// CHECK-LABEL: @dispatch_fn_a
util.func private @dispatch_fn_a() -> tensor<4xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %0 = flow.tensor.constant dense<123> : tensor<4xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %1 = flow.tensor.transfer %0 : tensor<4xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.call @dispatch_fn_b
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %2 = util.call @dispatch_fn_b(%1) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: util.call @dispatch_fn_c
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_c>]]
  %3 = util.call @dispatch_fn_c(%1) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %4 = flow.tensor.transfer %2 : tensor<4xi32> to #hal.device.promise<@dev_a>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_c>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %5 = flow.tensor.transfer %3 : tensor<4xi32> to #hal.device.promise<@dev_a>
  // CHECK: flow.dispatch @dispatch_a
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %6 = flow.dispatch @dispatch_a(%4, %5) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %5 : tensor<4xi32>
}
// CHECK: util.func private @dispatch_fn_b
util.func private @dispatch_fn_b(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %0 = flow.tensor.transfer %arg0 : tensor<4xi32> to #hal.device.promise<@dev_b>
  // CHECK: flow.dispatch @dispatch_b
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %1 = flow.dispatch @dispatch_b(%0) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  util.return %1 : tensor<4xi32>
}
// CHECK: util.func private @dispatch_fn_c
util.func private @dispatch_fn_c(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_c>]]
  %0 = flow.tensor.transfer %arg0 : tensor<4xi32> to #hal.device.promise<@dev_c>
  // CHECK: flow.dispatch @dispatch_c
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_c>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_c>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_c>]]
  %1 = flow.dispatch @dispatch_c(%0) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_c>]]
  util.return %1 : tensor<4xi32>
}

// -----

// Tests that consumer-placed ops are tracked across branch edges.

// CHECK-LABEL: @cfg_branch_constant_consumed
util.func private @cfg_branch_constant_consumed() -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: cf.br ^bb1
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  cf.br ^bb1(%cst : tensor<1xi32>)
^bb1(%bb1_arg0: tensor<1xi32>):
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.transfer %bb1_arg0 : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %cst_a : tensor<1xi32>
}

// -----

// Tests that producer-placed ops are tracked across branch edges.

// CHECK-LABEL: @cfg_branch_dispatch_produced
util.func private @cfg_branch_dispatch_produced() -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: cf.br ^bb1
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  cf.br ^bb1(%cst_a : tensor<1xi32>)
^bb1(%bb1_arg0: tensor<1xi32>):
  // CHECK: flow.dispatch @dispatch
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %dispatch_a = flow.dispatch @dispatch(%bb1_arg0) : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %dispatch_a : tensor<1xi32>
}

// -----

// Tests that back edges on loops track affinity changes.

// CHECK-LABEL: @cfg_loop_back_edge
util.func private @cfg_loop_back_edge() -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: cf.br ^bb1
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  cf.br ^bb1(%cst_a : tensor<1xi32>)
^bb1(%bb1_arg0: tensor<1xi32>):
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %bb1_arg0_b = flow.tensor.transfer %bb1_arg0 : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: util.call @step
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  %cond = util.call @step(%bb1_arg0_b) : (tensor<1xi32>) -> i1
  // CHECK: cf.cond_br
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_b>]]
  cf.cond_br %cond, ^bb1(%bb1_arg0 : tensor<1xi32>), ^bb2(%bb1_arg0_b : tensor<1xi32>)
^bb2(%bb2_arg0: tensor<1xi32>):
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_c>]]
  %bb2_arg0_c = flow.tensor.transfer %bb2_arg0 : tensor<1xi32> to #hal.device.promise<@dev_c>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_c>]]
  util.return %bb2_arg0_c : tensor<1xi32>
}
util.func private @step(tensor<1xi32>) -> i1

// -----

// Tests that conditional branches acting as selects propagate both affinities.

// CHECK-LABEL: @cfg_cond_branch_select
util.func private @cfg_cond_branch_select(%cond: i1) -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %cst_b = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_b>} dense<456> : tensor<1xi32>
  // CHECK: cf.cond_br
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_b>]]
  cf.cond_br %cond, ^bb1(%cst_a : tensor<1xi32>), ^bb1(%cst_b : tensor<1xi32>)
^bb1(%bb1_arg0: tensor<1xi32>):
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  util.return %bb1_arg0 : tensor<1xi32>
}

// -----

// Tests that consumer-placed ops through conditional branches acting as selects
// get placed on all targets.

// CHECK-LABEL: @cfg_cond_branch_select_consumer
util.func private @cfg_cond_branch_select_consumer(%cond: i1) -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: cf.cond_br
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>], [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  cf.cond_br %cond, ^bb1(%cst : tensor<1xi32>), ^bb2(%cst : tensor<1xi32>)
^bb1(%bb1_arg0: tensor<1xi32>):
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.transfer %bb1_arg0 : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %cst_a : tensor<1xi32>
^bb2(%bb2_arg0: tensor<1xi32>):
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %cst_b = flow.tensor.transfer %bb2_arg0 : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  util.return %cst_b : tensor<1xi32>
}

// -----

// Tests scf.if capturing consumer-placed ops tracks the affinity into nested
// regions.

// CHECK-LABEL: @scf_if_capture_consumer
util.func private @scf_if_capture_consumer(%cond: i1) -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: scf.if
  %cst_ab = scf.if %cond -> tensor<1xi32> {
    // CHECK: flow.tensor.transfer
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %cst_a = flow.tensor.transfer %cst : tensor<1xi32> to #hal.device.promise<@dev_a>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    scf.yield %cst_a : tensor<1xi32>
  // CHECK: else
  } else {
    // CHECK: flow.tensor.transfer
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
    %cst_b = flow.tensor.transfer %cst : tensor<1xi32> to #hal.device.promise<@dev_b>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
    scf.yield %cst_b : tensor<1xi32>
  // CHECK{LITERAL}: } {
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  }
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  util.return %cst_ab : tensor<1xi32>
}

// -----

// Tests scf.if capturing explicitly placed ops tracks the affinity of their
// produced results into consumers.

// CHECK-LABEL: @scf_if_capture_producer
util.func private @scf_if_capture_producer(%cond: i1) -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: scf.if
  %cst_bc = scf.if %cond -> tensor<1xi32> {
    // CHECK: flow.tensor.transfer
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
    %cst_b = flow.tensor.transfer %cst_a : tensor<1xi32> to #hal.device.promise<@dev_b>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
    scf.yield %cst_b : tensor<1xi32>
  // CHECK: else
  } else {
    // CHECK: flow.tensor.transfer
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_c>]]
    %cst_c = flow.tensor.transfer %cst_a : tensor<1xi32> to #hal.device.promise<@dev_c>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_c>]]
    scf.yield %cst_c : tensor<1xi32>
  // CHECK{LITERAL}: } {
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>, #hal.device.promise<@dev_c>]]
  }
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>, #hal.device.promise<@dev_c>]]
  util.return %cst_bc : tensor<1xi32>
}

// -----

// Tests scf.if returning unassigned consumer-placed operations has the affinity
// tracked across scf.yields and assigned based on the consumer.

// CHECK-LABEL: @scf_if_consumer_yield
util.func private @scf_if_consumer_yield(%cond: i1) -> tensor<1xi32> {
  // CHECK: scf.if
  %cst = scf.if %cond -> tensor<1xi32> {
    // CHECK: flow.tensor.constant
    // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %cst_0 = flow.tensor.constant dense<123> : tensor<1xi32>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    scf.yield %cst_0 : tensor<1xi32>
  // CHECK: else
  } else {
    // CHECK: flow.tensor.constant
    // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %cst_1 = flow.tensor.constant dense<456> : tensor<1xi32>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    scf.yield %cst_1 : tensor<1xi32>
  // CHECK{LITERAL}: } {
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  }
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.transfer %cst : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %cst_a : tensor<1xi32>
}

// -----

// Tests that consumer-placed ops get placed based on their use in the body.

// CHECK-LABEL: @scf_for_consumer_body_transfer
util.func private @scf_for_consumer_body_transfer() -> tensor<1xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: scf.for
  %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg0 = %cst) -> tensor<1xi32> {
    // CHECK: flow.tensor.transfer
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %arg0_a = flow.tensor.transfer %arg0 : tensor<1xi32> to #hal.device.promise<@dev_a>
    // CHECK: flow.dispatch @dispatch
    // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %t = flow.dispatch @dispatch(%arg0_a) : (tensor<1xi32>) -> tensor<1xi32>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    scf.yield %t : tensor<1xi32>
  // CHECK{LITERAL}: } {stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  }
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %for : tensor<1xi32>
}

// -----

// Tests that scf.for ops with transfers/explicit affinities on the edges get
// the

// CHECK-LABEL: @scf_for_boundary_transfer
util.func private @scf_for_boundary_transfer() -> (tensor<1xi32>, tensor<1xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: scf.for
  %for:2 = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg0 = %cst, %arg1 = %cst) -> (tensor<1xi32>, tensor<1xi32>) {
    // CHECK: flow.tensor.transfer
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %arg0_a = flow.tensor.transfer %arg0 : tensor<1xi32> to #hal.device.promise<@dev_a>
    // CHECK: flow.dispatch @dispatch
    // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %t = flow.dispatch @dispatch(%arg0_a) : (tensor<1xi32>) -> tensor<1xi32>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
    scf.yield %t, %arg1 : tensor<1xi32>, tensor<1xi32>
  // CHECK{LITERAL}: } {stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>], [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  }
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %for_0_b = flow.tensor.transfer %for#0 : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %for_1_b = flow.tensor.transfer %for#1 : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>], [#hal.device.promise<@dev_b>]]
  util.return %for_0_b, %for_1_b : tensor<1xi32>, tensor<1xi32>
}

// -----

// Tests that transfers track through iter_args.

// CHECK-LABEL: @scf_for_body_transfer
util.func private @scf_for_body_transfer() -> tensor<1xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: scf.for
  %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg0 = %cst_a) -> tensor<1xi32> {
    // CHECK: flow.tensor.transfer
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
    %arg0_b = flow.tensor.transfer %arg0 : tensor<1xi32> to #hal.device.promise<@dev_b>
    // CHECK: flow.dispatch @dispatch
    // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
    %t = flow.dispatch @dispatch(%arg0_b) : (tensor<1xi32>) -> tensor<1xi32>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
    scf.yield %t : tensor<1xi32>
  // CHECK{LITERAL}: } {stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  }
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_c>]]
  %for_c = flow.tensor.transfer %for : tensor<1xi32> to #hal.device.promise<@dev_c>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_c>]]
  util.return %for_c : tensor<1xi32>
}

// -----

// Tests that placed values track through iter_args to consumers in scf.for
// bodies.

// CHECK-LABEL: @scf_for_capture_producer
util.func private @scf_for_capture_producer() -> tensor<1xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: scf.for
  %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg0 = %cst_a) -> tensor<1xi32> {
    // CHECK: flow.dispatch @dispatch
    // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %t = flow.dispatch @dispatch(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    scf.yield %t : tensor<1xi32>
  // CHECK{LITERAL}: } {stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  }
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %for : tensor<1xi32>
}

// -----

// Tests that consumer-placed ops get placed based on their use in the body.

// CHECK-LABEL: @scf_while_consumer_body_transfer
util.func private @scf_while_consumer_body_transfer() -> tensor<1xi32> {
  %c0 = arith.constant 0 : index
  %c2_i32 = arith.constant 2 : i32
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: scf.while
  %while = scf.while(%arg0 = %cst) : (tensor<1xi32>) -> tensor<1xi32> {
    // CHECK: flow.tensor.load
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    %cond_i32 = flow.tensor.load %arg0[%c0] : tensor<1xi32>
    %cond = arith.cmpi slt, %cond_i32, %c2_i32 : i32
    // CHECK: scf.condition
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    scf.condition(%cond) %arg0 : tensor<1xi32>
  } do {
  ^bb0(%arg0: tensor<1xi32>):
    // CHECK: flow.tensor.transfer
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %arg0_a = flow.tensor.transfer %arg0 : tensor<1xi32> to #hal.device.promise<@dev_a>
    // CHECK: flow.dispatch @dispatch
    // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %t = flow.dispatch @dispatch(%arg0_a) : (tensor<1xi32>) -> tensor<1xi32>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    scf.yield %t : tensor<1xi32>
  // CHECK: } attributes {
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  }
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %while : tensor<1xi32>
}

// -----

// Tests that consumer-placed ops get placed based on their use as the result
// of an scf.while body.

// CHECK-LABEL: @scf_while_consumer_result_transfer
util.func private @scf_while_consumer_result_transfer() -> tensor<1xi32> {
  %c0 = arith.constant 0 : index
  %c2_i32 = arith.constant 2 : i32
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst = flow.tensor.constant dense<123> : tensor<1xi32>
  // CHECK: scf.while
  %while = scf.while(%arg0 = %cst) : (tensor<1xi32>) -> tensor<1xi32> {
    // CHECK: flow.tensor.load
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    %cond_i32 = flow.tensor.load %arg0[%c0] : tensor<1xi32>
    %cond = arith.cmpi slt, %cond_i32, %c2_i32 : i32
    // CHECK: scf.condition
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    scf.condition(%cond) %arg0 : tensor<1xi32>
  } do {
  ^bb0(%arg0: tensor<1xi32>):
    // CHECK: flow.dispatch @dispatch
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %t = flow.dispatch @dispatch(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    scf.yield %t : tensor<1xi32>
  // CHECK: } attributes {
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  }
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %while_a = flow.tensor.transfer %while : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %while_a : tensor<1xi32>
}

// -----

// Tests that transfers track through scf.while bodies.

// CHECK-LABEL: @scf_while_body_transfer
util.func private @scf_while_body_transfer() -> tensor<1xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: scf.while
  %while = scf.while(%arg0 = %cst_a) : (tensor<1xi32>) -> tensor<1xi32> {
    // CHECK: flow.tensor.load
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
    %cond_i32 = flow.tensor.load %arg0[%c0] : tensor<1xi32>
    %cond = arith.cmpi slt, %cond_i32, %c2_i32 : i32
    // CHECK: scf.condition
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
    scf.condition(%cond) %arg0 : tensor<1xi32>
  } do {
  ^bb0(%arg0: tensor<1xi32>):
    // CHECK: flow.tensor.transfer
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
    %arg0_b = flow.tensor.transfer %arg0 : tensor<1xi32> to #hal.device.promise<@dev_b>
    // CHECK: flow.dispatch @dispatch
    // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
    %t = flow.dispatch @dispatch(%arg0_b) : (tensor<1xi32>) -> tensor<1xi32>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
    scf.yield %t : tensor<1xi32>
  // CHECK: } attributes {
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  }
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_c>]]
  %while_c = flow.tensor.transfer %while : tensor<1xi32> to #hal.device.promise<@dev_c>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_c>]]
  util.return %while_c : tensor<1xi32>
}

// -----

// Tests that placed values track through to consumers in scf.while conditions.

// CHECK-LABEL: @scf_while_capture_producer_condition
util.func private @scf_while_capture_producer_condition() -> tensor<1xi32> {
  %c0 = arith.constant 0 : index
  %c2_i32 = arith.constant 2 : i32
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: scf.while
  %while = scf.while(%arg0 = %cst_a) : (tensor<1xi32>) -> tensor<1xi32> {
    // CHECK: flow.tensor.transfer
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %arg0_a = flow.tensor.transfer %arg0 : tensor<1xi32> to #hal.device.promise<@dev_a>
    // CHECK: flow.tensor.load
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    %cond_i32 = flow.tensor.load %arg0_a[%c0] : tensor<1xi32>
    %cond = arith.cmpi slt, %cond_i32, %c2_i32 : i32
    // CHECK: scf.condition
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    scf.condition(%cond) %arg0 : tensor<1xi32>
  } do {
  ^bb0(%arg0: tensor<1xi32>):
    // CHECK: flow.dispatch @dispatch
    // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %t = flow.dispatch @dispatch(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    scf.yield %t : tensor<1xi32>
  // CHECK: } attributes {
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  }
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %while : tensor<1xi32>
}

// -----

// Tests that placed values track through to consumers in scf.while bodies.

// CHECK-LABEL: @scf_while_capture_producer_body
util.func private @scf_while_capture_producer_body() -> tensor<1xi32> {
  %c0 = arith.constant 0 : index
  %c2_i32 = arith.constant 2 : i32
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst_a = flow.tensor.constant {stream.affinity = #hal.device.promise<@dev_a>} dense<123> : tensor<1xi32>
  // CHECK: scf.while
  %while = scf.while(%arg0 = %cst_a) : (tensor<1xi32>) -> tensor<1xi32> {
    // CHECK: flow.tensor.load
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    %cond_i32 = flow.tensor.load %arg0[%c0] : tensor<1xi32>
    %cond = arith.cmpi slt, %cond_i32, %c2_i32 : i32
    // CHECK: scf.condition
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    scf.condition(%cond) %arg0 : tensor<1xi32>
  } do {
  ^bb0(%arg0: tensor<1xi32>):
    // CHECK: flow.dispatch @dispatch
    // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    %t = flow.dispatch @dispatch(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
    // CHECK: scf.yield
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    scf.yield %t : tensor<1xi32>
  // CHECK: } attributes {
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  }
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %while : tensor<1xi32>
}

// -----

// Tests a realistic program with ABI ops.

// CHECK-LABEL: @simple_program
util.func public @simple_program(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view {
  // CHECK: hal.tensor.import
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %0 = hal.tensor.import on(#hal.device.promise<@dev_a>) wait(%arg1) => %arg0 "input0" : !hal.buffer_view -> tensor<1xi32>
  // CHECK: util.call @_simple_program
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %1 = util.call @_simple_program(%0) : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %2 = flow.tensor.transfer %1 : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK: hal.tensor.barrier
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %3 = hal.tensor.barrier join(%2 : tensor<1xi32>) => %arg2 : !hal.fence
  // CHECK: hal.tensor.export
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  %4 = hal.tensor.export on(#hal.device.promise<@dev_a>) %3 "output0" : tensor<1xi32> -> !hal.buffer_view
  util.return %4 : !hal.buffer_view
}
// CHECK: util.func private @_simple_program
util.func private @_simple_program(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: util.call @dispatch_a
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %0 = util.call @dispatch_a(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_b>]]
  %1 = flow.tensor.transfer %0 : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK: util.call @dispatch_b
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %2 = util.call @dispatch_b(%1) : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  util.return %2 : tensor<1xi32>
}
// CHECK: util.func private @dispatch_a
util.func private @dispatch_a(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  %cst = flow.tensor.constant dense<[1]> : tensor<1xi32>
  // CHECK: flow.dispatch @dispatch_a
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>], [#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %0 = flow.dispatch @dispatch_a(%arg0, %cst) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  util.return %0 : tensor<1xi32>
}
// CHECK: util.func private @dispatch_b
util.func private @dispatch_b(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  %cst = flow.tensor.constant dense<[2]> : tensor<1xi32>
  // CHECK: flow.dispatch @dispatch_b
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>], [#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %0 = flow.dispatch @dispatch_b(%arg0, %cst) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  util.return %0 : tensor<1xi32>
}

// -----

// Tests that a value returned from a stream.yield takes on the consumer
// affinities of the parent region.

// CHECK-LABEL: @stream_yield_consumer_affinity
util.func public @stream_yield_consumer_affinity(%size: index) -> !stream.resource<transient> {
  %input = stream.async.constant on(#hal.device.promise<@dev_a>) : !stream.resource<transient>{%size} = dense<3> : tensor<8xi32>
  // CHECK: stream.async.execute
  %result_a, %timepoint_a = stream.async.execute on(#hal.device.promise<@dev_a>) with(%input as %capture_a: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    // CHECK: stream.async.clone
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
    %t_a = stream.async.clone %capture_a : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size}
    stream.yield %t_a : !stream.resource<transient>{%size}
  // CHECK{LITERAL}: => !stream.timepoint attributes {
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  } => !stream.timepoint
  // CHECK: stream.async.execute
  %result_b, %timepoint_b = stream.async.execute on(#hal.device.promise<@dev_b>) await(%timepoint_a) => with(%result_a as %capture_b: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    %t_b = stream.async.clone %capture_b : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size}
    stream.yield %t_b : !stream.resource<transient>{%size}
  // CHECK{LITERAL}: => !stream.timepoint attributes {
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_b>]]
  } => !stream.timepoint
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  util.return %result_b : !stream.resource<transient>
}

// -----

// Tests that a tied value through an execution region has its producer/consumer
// affinity assigned based on subsequent usage.

// CHECK-LABEL: @stream_yield_consumer_affinity_tied
util.func public @stream_yield_consumer_affinity_tied(%size: index) -> !stream.resource<transient> {
  %c0 = arith.constant 0 : index
  %input = stream.async.constant on(#hal.device.promise<@dev_a>) : !stream.resource<transient>{%size} = dense<3> : tensor<8xi32>
  // CHECK: stream.async.execute
  %result_a, %timepoint_a = stream.async.execute on(#hal.device.promise<@dev_a>) with(%input as %capture_a: !stream.resource<transient>{%size}) -> %input{%size} {
    // CHECK: stream.async.dispatch
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
    %t_a = stream.async.dispatch @executable::@dispatch(%capture_a[%c0 to %size for %size]) : (!stream.resource<transient>{%size}) -> %capture_a{%size}
    stream.yield %t_a : !stream.resource<transient>{%size}
  // CHECK{LITERAL}: => !stream.timepoint attributes {
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  } => !stream.timepoint
  // CHECK: stream.async.execute
  %result_b, %timepoint_b = stream.async.execute on(#hal.device.promise<@dev_b>) await(%timepoint_a) => with(%result_a as %capture_b: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    // CHECK: stream.async.clone
    // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
    // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_b>]]
    %t_b = stream.async.clone %capture_b : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size}
    stream.yield %t_b : !stream.resource<transient>{%size}
  // CHECK{LITERAL}: => !stream.timepoint attributes {
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_b>]]
  } => !stream.timepoint
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  util.return %result_b : !stream.resource<transient>
}

// -----

// Tests that pinning a value will always result in a valid analysis result even
// if we couldn't exhaustively analyze the value. Here %arg0 has producers (in
// the callers) and consumers (in the callers after return) we don't know about
// and would otherwise mark the value has having an unknown affinity.
// Note that %arg0 has no operand affinity/usage - we can only make assertions
// about the result.

// CHECK-LABEL: @pinning_overrides_invalid
util.func public @pinning_overrides_invalid(%arg0: tensor<128xi8>) -> tensor<128xi8> {
  // CHECK: flow.tensor.barrier
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@device>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@device>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@device>]]
  %0 = flow.tensor.barrier %arg0 : tensor<128xi8> on #hal.device.promise<@device>
  // CHECK: flow.dispatch
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@device>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@device>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@device>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@device>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@device>]]
  %1 = flow.dispatch @ex::@entry(%0) : (tensor<128xi8>) -> tensor<128xi8>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@device>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@device>]]
  util.return %1 : tensor<128xi8>
}

// -----

// Tests that pinning can be used to control I/O to external methods.

util.func private @extern(tensor<128xi8>) -> tensor<128xi8>

// CHECK-LABEL: @pinning_overrides_invalid_call
util.func public @pinning_overrides_invalid_call() -> tensor<128xi8> {
  // CHECK: %[[CST:.+]] = flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %cst = flow.tensor.constant dense<123> : tensor<128xi8>
  // CHECK: flow.tensor.barrier %[[CST]]
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %cst_barrier = flow.tensor.barrier %cst : tensor<128xi8> on #hal.device.promise<@dev_a>
  // CHECK: %[[RESULT:.+]] = util.call
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_b>]]
  %result = util.call @extern(%cst_barrier) : (tensor<128xi8>) -> tensor<128xi8>
  // CHECK: flow.tensor.barrier %[[RESULT]]
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_b>]]
  %result_barrier = flow.tensor.barrier %result : tensor<128xi8> on #hal.device.promise<@dev_b>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_b>]]
  util.return %result_barrier : tensor<128xi8>
}

// -----

// Tests that very long chains of tied ops still resolve.
// This is a stress test for the solver; with the way it works and our elements
// are defined each level of the dependency DAG requires one iteration to solve.
// If you have 100 levels you need 100 iterations. The real problem is needing
// 100 levels, but solving that is harder.

// CHECK-LABEL: @long_tied_chain_pinned_bottom
util.func public @long_tied_chain_pinned_bottom() -> tensor<4xi32> {
  // CHECK: flow.tensor.constant
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_b>]]
  %cst = flow.tensor.constant dense<123> : tensor<4xi32>
  // CHECK: flow.tensor.transfer
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_b>]]
  %cst_b = flow.tensor.transfer %cst : tensor<4xi32> to #hal.device.promise<@dev_b>
  // CHECK: flow.dispatch @dispatch0
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %t0 = flow.dispatch @dispatch0(%cst_b) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: flow.dispatch @dispatch1
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %t1 = flow.dispatch @dispatch1(%t0) : (tensor<4xi32>) -> %t0
  %t2 = flow.dispatch @dispatch2(%t1) : (tensor<4xi32>) -> %t1
  %t3 = flow.dispatch @dispatch3(%t2) : (tensor<4xi32>) -> %t2
  %t4 = flow.dispatch @dispatch4(%t3) : (tensor<4xi32>) -> %t3
  %t5 = flow.dispatch @dispatch5(%t4) : (tensor<4xi32>) -> %t4
  %t6 = flow.dispatch @dispatch6(%t5) : (tensor<4xi32>) -> %t5
  %t7 = flow.dispatch @dispatch7(%t6) : (tensor<4xi32>) -> %t6
  %t8 = flow.dispatch @dispatch8(%t7) : (tensor<4xi32>) -> %t7
  %t9 = flow.dispatch @dispatch9(%t8) : (tensor<4xi32>) -> %t8
  %t10 = flow.dispatch @dispatch10(%t9) : (tensor<4xi32>) -> %t9
  %t11 = flow.dispatch @dispatch11(%t10) : (tensor<4xi32>) -> %t10
  %t12 = flow.dispatch @dispatch12(%t11) : (tensor<4xi32>) -> %t11
  %t13 = flow.dispatch @dispatch13(%t12) : (tensor<4xi32>) -> %t12
  %t14 = flow.dispatch @dispatch14(%t13) : (tensor<4xi32>) -> %t13
  %t15 = flow.dispatch @dispatch15(%t14) : (tensor<4xi32>) -> %t14
  %t16 = flow.dispatch @dispatch16(%t15) : (tensor<4xi32>) -> %t15
  %t17 = flow.dispatch @dispatch17(%t16) : (tensor<4xi32>) -> %t16
  %t18 = flow.dispatch @dispatch18(%t17) : (tensor<4xi32>) -> %t17
  %t19 = flow.dispatch @dispatch19(%t18) : (tensor<4xi32>) -> %t18
  %t20 = flow.dispatch @dispatch20(%t19) : (tensor<4xi32>) -> %t19
  %t21 = flow.dispatch @dispatch21(%t20) : (tensor<4xi32>) -> %t20
  // CHECK: flow.dispatch @dispatch22
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_b>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %t22 = flow.dispatch @dispatch22(%t21) : (tensor<4xi32>) -> %t21
  // CHECK: flow.tensor.barrier
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  %barrier = flow.tensor.barrier %t22 : tensor<4xi32> on #hal.device.promise<@dev_a>
  // CHECK: util.return
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>, #hal.device.promise<@dev_b>]]
  util.return %barrier : tensor<4xi32>
}

// -----

// Similar to @long_tied_chain_pinned_top above but pinning at the top of the
// use-def chain.

// CHECK-LABEL: @long_tied_chain_pinned_top
util.func public @long_tied_chain_pinned_top(%buffer_view: !hal.buffer_view) {
  %storage = hal.tensor.import on(#hal.device.promise<@dev_a>) %buffer_view "input" : !hal.buffer_view -> tensor<4xi32>
  // CHECK: flow.dispatch @dispatch0
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %t0 = flow.dispatch @dispatch0(%storage) : (tensor<4xi32>) -> %storage
  %t1 = flow.dispatch @dispatch1(%t0) : (tensor<4xi32>) -> %t0
  %t2 = flow.dispatch @dispatch2(%t1) : (tensor<4xi32>) -> %t1
  %t3 = flow.dispatch @dispatch3(%t2) : (tensor<4xi32>) -> %t2
  %t4 = flow.dispatch @dispatch4(%t3) : (tensor<4xi32>) -> %t3
  %t5 = flow.dispatch @dispatch5(%t4) : (tensor<4xi32>) -> %t4
  %t6 = flow.dispatch @dispatch6(%t5) : (tensor<4xi32>) -> %t5
  %t7 = flow.dispatch @dispatch7(%t6) : (tensor<4xi32>) -> %t6
  %t8 = flow.dispatch @dispatch8(%t7) : (tensor<4xi32>) -> %t7
  %t9 = flow.dispatch @dispatch9(%t8) : (tensor<4xi32>) -> %t8
  %t10 = flow.dispatch @dispatch10(%t9) : (tensor<4xi32>) -> %t9
  %t11 = flow.dispatch @dispatch11(%t10) : (tensor<4xi32>) -> %t10
  %t12 = flow.dispatch @dispatch12(%t11) : (tensor<4xi32>) -> %t11
  %t13 = flow.dispatch @dispatch13(%t12) : (tensor<4xi32>) -> %t12
  %t14 = flow.dispatch @dispatch14(%t13) : (tensor<4xi32>) -> %t13
  %t15 = flow.dispatch @dispatch15(%t14) : (tensor<4xi32>) -> %t14
  %t16 = flow.dispatch @dispatch16(%t15) : (tensor<4xi32>) -> %t15
  %t17 = flow.dispatch @dispatch17(%t16) : (tensor<4xi32>) -> %t16
  %t18 = flow.dispatch @dispatch18(%t17) : (tensor<4xi32>) -> %t17
  %t19 = flow.dispatch @dispatch19(%t18) : (tensor<4xi32>) -> %t18
  %t20 = flow.dispatch @dispatch20(%t19) : (tensor<4xi32>) -> %t19
  %t21 = flow.dispatch @dispatch21(%t20) : (tensor<4xi32>) -> %t20
  // CHECK: flow.dispatch @dispatch22
  // CHECK-SAME{LITERAL}: stream.affinities = [#hal.device.promise<@dev_a>]
  // CHECK-SAME{LITERAL}: stream.affinities.operands = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.operands.usage = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results = [[#hal.device.promise<@dev_a>]]
  // CHECK-SAME{LITERAL}: stream.affinities.results.usage = [[#hal.device.promise<@dev_a>]]
  %t22 = flow.dispatch @dispatch22(%t21) : (tensor<4xi32>) -> %t21
  util.return
}
