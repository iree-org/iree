// RUN: iree-opt --split-input-file --iree-stream-propagate-timepoints %s | FileCheck %s

// Tests that PropagateTimepoints properly handles SCF operations via MutableRegionBranchOpInterface.

// CHECK-LABEL: @propagate_if
util.func public @propagate_if(%cond: i1, %arg: !stream.resource<external>) -> !stream.resource<external> {
  %c8 = arith.constant 8 : index
  // CHECK: %[[IF_RESULT:.+]]:2 = scf.if %{{.*}} -> (!stream.resource<external>, !stream.timepoint)
  %0 = scf.if %cond -> !stream.resource<external> {
    %1, %2 = stream.async.execute with(%arg as %arg2: !stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} {
      %3 = stream.async.clone %arg2 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}
      stream.yield %3 : !stream.resource<external>{%c8}
    } => !stream.timepoint
    %4 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c8}
    // CHECK: scf.yield %{{.*}}, %{{.*}} : !stream.resource<external>, !stream.timepoint
    scf.yield %4 : !stream.resource<external>
  } else {
    // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
    // CHECK: scf.yield %{{.*}}, %[[IMM]] : !stream.resource<external>, !stream.timepoint
    scf.yield %arg : !stream.resource<external>
  }
  // CHECK: %[[AWAIT:.+]] = stream.timepoint.await %[[IF_RESULT]]#1 => %[[IF_RESULT]]#0
  // CHECK: util.return %[[AWAIT]]
  util.return %0 : !stream.resource<external>
}

// -----

// CHECK-LABEL: @propagate_for
util.func public @propagate_for(%arg: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  // CHECK: %[[INIT_TP:.+]] = stream.timepoint.immediate
  // CHECK: %[[FOR_RESULT:.+]]:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
  // CHECK-SAME: iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %[[INIT_TP]])
  // CHECK-SAME: -> (!stream.resource<external>, !stream.timepoint)
  %0 = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter = %arg) -> !stream.resource<external> {
    %1, %2 = stream.async.execute with(%iter as %arg2: !stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} {
      %3 = stream.async.clone %arg2 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}
      stream.yield %3 : !stream.resource<external>{%c8}
    } => !stream.timepoint
    %4 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c8}
    // CHECK: scf.yield %{{.*}}, %{{.*}} : !stream.resource<external>, !stream.timepoint
    scf.yield %4 : !stream.resource<external>
  }
  // CHECK: %[[AWAIT:.+]] = stream.timepoint.await %[[FOR_RESULT]]#1 => %[[FOR_RESULT]]#0
  // CHECK: util.return %[[AWAIT]]
  util.return %0 : !stream.resource<external>
}

// -----

// CHECK-LABEL: @propagate_while
util.func public @propagate_while(%arg: !stream.resource<external>) -> !stream.resource<external> {
  %c8 = arith.constant 8 : index
  %true = arith.constant true
  %false = arith.constant false

  // CHECK: %[[INIT_TP:.+]] = stream.timepoint.immediate
  // CHECK: %[[WHILE_RESULT:.+]]:2 = scf.while
  // CHECK-SAME: (%{{.*}} = %{{.*}}, %{{.*}} = %[[INIT_TP]])
  // CHECK-SAME: : (!stream.resource<external>, !stream.timepoint)
  // CHECK-SAME: -> (!stream.resource<external>, !stream.timepoint)
  %0 = scf.while (%arg0 = %arg) : (!stream.resource<external>) -> !stream.resource<external> {
    // CHECK: scf.condition(%{{.*}}) %{{.*}}, %{{.*}} : !stream.resource<external>, !stream.timepoint
    scf.condition(%true) %arg0 : !stream.resource<external>
  } do {
  ^bb0(%arg1: !stream.resource<external>):
    %1, %2 = stream.async.execute with(%arg1 as %arg2: !stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} {
      %3 = stream.async.clone %arg2 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}
      stream.yield %3 : !stream.resource<external>{%c8}
    } => !stream.timepoint
    %4 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c8}
    // CHECK: scf.yield %{{.*}}, %{{.*}} : !stream.resource<external>, !stream.timepoint
    scf.yield %4 : !stream.resource<external>
  }
  // CHECK: %[[AWAIT:.+]] = stream.timepoint.await %[[WHILE_RESULT]]#1 => %[[WHILE_RESULT]]#0
  // CHECK: util.return %[[AWAIT]]
  util.return %0 : !stream.resource<external>
}

// -----

// CHECK-LABEL: @propagate_index_switch
util.func public @propagate_index_switch(%idx: index, %arg: !stream.resource<external>) -> !stream.resource<external> {
  %c8 = arith.constant 8 : index

  // CHECK: %[[SWITCH_RESULT:.+]]:2 = scf.index_switch %{{.*}} -> (!stream.resource<external>, !stream.timepoint)
  %0 = scf.index_switch %idx -> !stream.resource<external>
  case 0 {
    %1, %2 = stream.async.execute with(%arg as %arg2: !stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} {
      %3 = stream.async.clone %arg2 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}
      stream.yield %3 : !stream.resource<external>{%c8}
    } => !stream.timepoint
    %4 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c8}
    // CHECK: scf.yield %{{.*}}, %{{.*}} : !stream.resource<external>, !stream.timepoint
    scf.yield %4 : !stream.resource<external>
  }
  case 1 {
    // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
    // CHECK: scf.yield %{{.*}}, %[[IMM]] : !stream.resource<external>, !stream.timepoint
    scf.yield %arg : !stream.resource<external>
  }
  default {
    // CHECK: %[[IMM_DEFAULT:.+]] = stream.timepoint.immediate
    // CHECK: scf.yield %{{.*}}, %[[IMM_DEFAULT]] : !stream.resource<external>, !stream.timepoint
    scf.yield %arg : !stream.resource<external>
  }
  // CHECK: %[[AWAIT:.+]] = stream.timepoint.await %[[SWITCH_RESULT]]#1 => %[[SWITCH_RESULT]]#0
  // CHECK: util.return %[[AWAIT]]
  util.return %0 : !stream.resource<external>
}

// -----

// CHECK-LABEL: @nested_if_for
util.func public @nested_if_for(%cond: i1, %arg: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  // CHECK: %[[IF_RESULT:.+]]:2 = scf.if %{{.*}} -> (!stream.resource<external>, !stream.timepoint)
  %0 = scf.if %cond -> !stream.resource<external> {
    // CHECK: %[[FOR_RESULT:.+]]:2 = scf.for
    %1 = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter = %arg) -> !stream.resource<external> {
      %2, %3 = stream.async.execute with(%iter as %arg2: !stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} {
        %4 = stream.async.clone %arg2 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}
        stream.yield %4 : !stream.resource<external>{%c8}
      } => !stream.timepoint
      %5 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c8}
      scf.yield %5 : !stream.resource<external>
    }
    // CHECK: %[[FOR_AWAIT:.+]] = stream.timepoint.await %[[FOR_RESULT]]#1 => %[[FOR_RESULT]]#0
    // CHECK: scf.yield %[[FOR_AWAIT]], %[[FOR_RESULT]]#1
    scf.yield %1 : !stream.resource<external>
  } else {
    scf.yield %arg : !stream.resource<external>
  }
  // CHECK: %[[FINAL_AWAIT:.+]] = stream.timepoint.await %[[IF_RESULT]]#1 => %[[IF_RESULT]]#0
  // CHECK: util.return %[[FINAL_AWAIT]]
  util.return %0 : !stream.resource<external>
}
