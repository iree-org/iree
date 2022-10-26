// RUN: iree-opt --split-input-file --iree-stream-propagate-timepoints %s | FileCheck %s

// Tests that resource global loads pull an unready resource and provide an
// await with the associated timepoint.
//
// This rotates waits through stores and into loads.

// CHECK: util.global private mutable @constantGlobal__timepoint = #stream.timepoint<immediate>
// CHECK-NEXT: util.global private mutable @constantGlobal : !stream.resource<constant>
util.global private mutable @constantGlobal : !stream.resource<constant>

// CHECK-LABEL: @globalLoad
func.func @globalLoad() {
  // CHECK-NEXT: %[[TIMEPOINT:.+]] = util.global.load @constantGlobal__timepoint : !stream.timepoint
  // CHECK-NEXT: %[[UNREADY:.+]] = util.global.load @constantGlobal : !stream.resource<constant>
  // CHECK-NEXT: %[[SIZE:.+]] = stream.resource.size %[[UNREADY]]
  // CHECK-NEXT: %[[VALUE:.+]] = stream.timepoint.await %[[TIMEPOINT]] => %[[UNREADY]] : !stream.resource<constant>{%[[SIZE]]}
  %0 = util.global.load @constantGlobal : !stream.resource<constant>
  // CHECK-NEXT: util.optimization_barrier %[[VALUE]]
  util.optimization_barrier %0 : !stream.resource<constant>
  return
}

// -----

// Tests that resource global stores consume their incoming timepoints.
// Here the function gets a timepoint + unready resource and we forward that
// directly into an expanded (timepoint, resource) global.
//
// This rotates waits through stores and into loads.

// CHECK: util.global private mutable @mutableGlobal__timepoint = #stream.timepoint<immediate>
// CHECK-NEXT: util.global private mutable @mutableGlobal : !stream.resource<variable>
util.global private mutable @mutableGlobal : !stream.resource<variable>

// CHECK-LABEL: @globalStore
// CHECK-SAME: (%[[TIMEPOINT:.+]]: !stream.timepoint, %[[UNREADY:.+]]: !stream.resource<variable>)
func.func private @globalStore(%arg0: !stream.resource<variable>) {
  //      CHECK: util.global.store %[[TIMEPOINT]], @mutableGlobal__timepoint : !stream.timepoint
  // CHECK-NEXT: util.global.store %[[UNREADY]], @mutableGlobal : !stream.resource<variable>
  util.global.store %arg0, @mutableGlobal : !stream.resource<variable>
  return
}

// -----

// Tests that function arguments are expanded into (timepoint, resource) and
// an await is inserted to tie them together.
//
// This rotates waits from callers into callees.

// CHECK-LABEL: @funcArgs
// CHECK-SAME: (%[[TIMEPOINT0:.+]]: !stream.timepoint, %[[UNREADY0:.+]]: !stream.resource<external>,
// CHECK-SAME:  %[[TIMEPOINT1:.+]]: !stream.timepoint, %[[UNREADY1:.+]]: !stream.resource<transient>)
func.func private @funcArgs(%arg0: !stream.resource<external>, %arg1: !stream.resource<transient>) {
  // CHECK-NEXT: %[[SIZE0:.+]] = stream.resource.size %[[UNREADY0]] : !stream.resource<external>
  // CHECK-NEXT: %[[READY0:.+]] = stream.timepoint.await %[[TIMEPOINT0]] => %[[UNREADY0]] : !stream.resource<external>{%[[SIZE0]]}
  // CHECK-NEXT: %[[SIZE1:.+]] = stream.resource.size %[[UNREADY1]] : !stream.resource<transient>
  // CHECK-NEXT: %[[READY1:.+]] = stream.timepoint.await %[[TIMEPOINT1]] => %[[UNREADY1]] : !stream.resource<transient>{%[[SIZE1]]}

  // CHECK-NEXT: util.optimization_barrier %[[READY0]]
  util.optimization_barrier %arg0 : !stream.resource<external>
  // CHECK-NEXT: util.optimization_barrier %[[READY1]]
  util.optimization_barrier %arg1 : !stream.resource<transient>
  return
}

// -----

// Tests that function results are expanded into (timepoint, resource) and
// awaits are consumed.
//
// This rotates waits from callees into callers.

// CHECK-LABEL: @funcResults
// CHECK-SAME: (%[[TIMEPOINT0:.+]]: !stream.timepoint, %[[UNREADY0:.+]]: !stream.resource<external>,
// CHECK-SAME:  %[[TIMEPOINT1:.+]]: !stream.timepoint, %[[UNREADY1:.+]]: !stream.resource<transient>)
func.func private @funcResults(%arg0: !stream.resource<external>, %arg1: !stream.resource<transient>) -> (!stream.resource<external>, !stream.resource<transient>) {
  // NOTE: there will be extra stuff here from the arg insertion. Since the
  // return should consume the await that was inserted we expect to directly use
  // the function arguments.

  // CHECK: return %[[TIMEPOINT0]], %[[UNREADY0]], %[[TIMEPOINT1]], %[[UNREADY1]]
  return %arg0, %arg1 : !stream.resource<external>, !stream.resource<transient>
}

// -----

// Tests that function calls have their args and results expanded into
// (timepoint, resource) and awaits are consumed on the arguments. Results will
// have awaits inserted.
//
// This rotates waits on args from callers to callees and waits on results from
// callees to callers.

// CHECK-LABEL: @caller
// CHECK-SAME: (%[[TIMEPOINT0:.+]]: !stream.timepoint, %[[UNREADY0:.+]]: !stream.resource<external>,
// CHECK-SAME:  %[[TIMEPOINT1:.+]]: !stream.timepoint, %[[UNREADY1:.+]]: !stream.resource<transient>)
func.func private @caller(%arg0: !stream.resource<external>, %arg1: !stream.resource<transient>) {
  // NOTE: there will be extra stuff here from the arg insertion. The call
  // consumes the unready resources and we expect the args to be passed
  // directly.

  // CHECK: %[[RET:.+]]:4 = call @callee(%[[TIMEPOINT0]], %[[UNREADY0]], %[[TIMEPOINT1]], %[[UNREADY1]])
  // CHECK-SAME: : (!stream.timepoint, !stream.resource<external>, !stream.timepoint, !stream.resource<transient>) -> (!stream.timepoint, !stream.resource<external>, !stream.timepoint, !stream.resource<transient>)
  %0:2 = call @callee(%arg0, %arg1) : (!stream.resource<external>, !stream.resource<transient>) -> (!stream.resource<external>, !stream.resource<transient>)
  // CHECK-NEXT: %[[RET_SIZE0:.+]] = stream.resource.size %[[RET]]#1 : !stream.resource<external>
  // CHECK-NEXT: %[[RET_READY0:.+]] = stream.timepoint.await %[[RET]]#0 => %[[RET]]#1 : !stream.resource<external>{%[[RET_SIZE0]]}
  // CHECK-NEXT: %[[RET_SIZE1:.+]] = stream.resource.size %[[RET]]#3 : !stream.resource<transient>
  // CHECK-NEXT: %[[RET_READY1:.+]] = stream.timepoint.await %[[RET]]#2 => %[[RET]]#3 : !stream.resource<transient>{%[[RET_SIZE1]]}

  // CHECK-NEXT: util.optimization_barrier %[[RET_READY0]] : !stream.resource<external>
  util.optimization_barrier %0#0 : !stream.resource<external>
  // CHECK-NEXT: util.optimization_barrier %[[RET_READY1]] : !stream.resource<transient>
  util.optimization_barrier %0#1 : !stream.resource<transient>

  return
}

func.func private @callee(%arg0: !stream.resource<external>, %arg1: !stream.resource<transient>) -> (!stream.resource<external>, !stream.resource<transient>) {
  return %arg0, %arg1 : !stream.resource<external>, !stream.resource<transient>
}

// -----

// Tests that branch args are expanded into (timepoint, resource) and that
// branch operands are properly expanded.
//
// This rotates waits on branch operands into successors.

// CHECK-LABEL: @br
// CHECK-SAME: (%[[TIMEPOINT0:.+]]: !stream.timepoint, %[[UNREADY0:.+]]: !stream.resource<external>,
// CHECK-SAME:  %[[TIMEPOINT1:.+]]: !stream.timepoint, %[[UNREADY1:.+]]: !stream.resource<transient>)
func.func private @br(%arg0: !stream.resource<external>, %arg1: !stream.resource<transient>) {
  // NOTE: there will be extra stuff here from the arg insertion. The branch
  // consumes the unready resources and we expect the args to be passed directly
  // to the cf.br.

  // CHECK: cf.br ^bb1(%[[TIMEPOINT0]], %[[UNREADY0]], %[[TIMEPOINT1]], %[[UNREADY1]]
  cf.br ^bb1(%arg0, %arg1 : !stream.resource<external>, !stream.resource<transient>)

// CHECK-NEXT: ^bb1(%[[BB1_TIMEPOINT0:.+]]: !stream.timepoint, %[[BB1_UNREADY0:.+]]: !stream.resource<external>,
// CHECK-SAME:      %[[BB1_TIMEPOINT1:.+]]: !stream.timepoint, %[[BB1_UNREADY1:.+]]: !stream.resource<transient>):
^bb1(%bb1_arg0: !stream.resource<external>, %bb1_arg1: !stream.resource<transient>):
  // CHECK-NEXT: %[[SIZE0:.+]] = stream.resource.size %[[BB1_UNREADY0]] : !stream.resource<external>
  // CHECK-NEXT: %[[READY0:.+]] = stream.timepoint.await %[[BB1_TIMEPOINT0]] => %[[BB1_UNREADY0]] : !stream.resource<external>{%8}
  // CHECK-NEXT: %[[SIZE1:.+]] = stream.resource.size %[[BB1_UNREADY1]] : !stream.resource<transient>
  // CHECK-NEXT: %[[READY1:.+]] = stream.timepoint.await %[[BB1_TIMEPOINT1]] => %[[BB1_UNREADY1]] : !stream.resource<transient>{%10}

  // CHECK-NEXT: util.optimization_barrier %[[READY0]]
  util.optimization_barrier %bb1_arg0 : !stream.resource<external>
  // CHECK-NEXT: util.optimization_barrier %[[READY1]]
  util.optimization_barrier %bb1_arg1 : !stream.resource<transient>
  return
}

// -----

// Tests that stream.async.execute consumes incoming timepoints.
// If multiple timepoints are required for the captures then a
// stream.timepoint.join should be emitted.
//
// This rotates waits on producers to waits on consumers.

// CHECK-LABEL: @asyncExecuteConsume
// CHECK-SAME: (%[[TIMEPOINT0:.+]]: !stream.timepoint, %[[UNREADY0:.+]]: !stream.resource<external>,
// CHECK-SAME:  %[[TIMEPOINT1:.+]]: !stream.timepoint, %[[UNREADY1:.+]]: !stream.resource<transient>)
func.func private @asyncExecuteConsume(%arg0: !stream.resource<external>, %arg1: !stream.resource<transient>) {
  // NOTE: there will be extra stuff here from the arg insertion. The execution
  // region consumes the unready resources and we expect the args to be captured
  // directly.

  %arg0_size = stream.resource.size %arg0 : !stream.resource<external>
  %arg1_size = stream.resource.size %arg1 : !stream.resource<transient>

  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[TIMEPOINT0]], %[[TIMEPOINT1]]) => !stream.timepoint
  // CHECK: = stream.async.execute await(%[[JOIN]])
  // CHECK-SAME: with(%[[UNREADY0]] as %{{.+}}: !stream.resource<external>{%{{[a-z0-9]+}}},
  // CHECK-SAME:      %[[UNREADY1]] as %{{.+}}: !stream.resource<transient>{%{{.+}}})
  %results:2, %results_timepoint = stream.async.execute
      with(%arg0 as %arg0_capture: !stream.resource<external>{%arg0_size},
           %arg1 as %arg1_capture: !stream.resource<transient>{%arg1_size})
      -> (!stream.resource<external>{%arg0_size}, !stream.resource<transient>{%arg1_size}) {
    stream.yield %arg0_capture, %arg1_capture : !stream.resource<external>{%arg0_size}, !stream.resource<transient>{%arg1_size}
  } => !stream.timepoint
  %ready_results:2 = stream.timepoint.await %results_timepoint => %results#0, %results#1 : !stream.resource<external>{%arg0_size}, !stream.resource<transient>{%arg1_size}
  util.optimization_barrier %ready_results#0 : !stream.resource<external>
  util.optimization_barrier %ready_results#1 : !stream.resource<transient>
  return
}
