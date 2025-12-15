// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-lower-structural-pcf, cse)" --split-input-file --verify-diagnostics | FileCheck %s

util.func private @generic() {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    util.optimization_barrier %id, %n : index, index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @generic
//   CHECK-DAG:   %[[I:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[N:.+]] = arith.constant 1 : index
//       CHECK:   scf.execute_region
//  CHECK-NEXT:     util.optimization_barrier %[[I]], %[[N]]
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }

// -----

util.func private @generic_multiple_iterators(%d0: index, %d1: index) {
  pcf.generic scope(#pcf.sequential)
    execute[%id0: index, %id1: index, %n0: index, %n1: index] {
    util.optimization_barrier %id0, %id1, %n0, %n1 : index, index, index, index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @generic_multiple_iterators
//   CHECK-DAG:   %[[I:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[N:.+]] = arith.constant 1 : index
//       CHECK:   scf.execute_region
//  CHECK-NEXT:     util.optimization_barrier %[[I]], %[[I]], %[[N]], %[[N]]
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }

// -----

util.func private @generic_with_multiple_blocks() {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    cf.br ^bb1
   ^bb1:
    util.optimization_barrier
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @generic_with_multiple_blocks
//       CHECK:   scf.execute_region
//  CHECK-NEXT:     cf.br ^bb1
//  CHECK-NEXT:    ^bb1:  // pred: ^bb0
//  CHECK-NEXT:     util.optimization_barrier
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }

// -----

util.func private @br_cond_return(%cond: i1) {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    pcf.br.cond_return %cond ^bb1
   ^bb1:
    util.optimization_barrier
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @br_cond_return
//  CHECK-SAME:  %[[COND:[A-Za-z0-9_]+]]: i1
//       CHECK:   scf.execute_region
//  CHECK-NEXT:     cf.cond_br %[[COND]], ^bb2, ^bb1

// Dest branch (condition is false).
//  CHECK-NEXT:    ^bb1:  // pred: ^bb0
//  CHECK-NEXT:     util.optimization_barrier
//  CHECK-NEXT:     scf.yield

// Return branch (condition is true).
//  CHECK-NEXT:    ^bb2:  // pred: ^bb0
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }

// -----

util.func private @fail_barrier_build(%cond: i1) {
// expected-error@+1 {{failed to construct requested barrier}}
  pcf.generic sync true scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    pcf.return
  }
  util.return
}

// -----

util.func private @lower_loop(%d0: index, %d1: index) {
  pcf.loop scope(#pcf.sequential) count(%d0, %d1)
    execute[%n0: index, %n1: index] {
    util.optimization_barrier %n0, %n1 : index, index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @lower_loop
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9_]+]]: index
//       CHECK:   scf.forall (%[[ID1:.+]], %[[ID0:.+]]) in (%[[ARG1]], %[[ARG0]])
//  CHECK-NEXT:     util.optimization_barrier %[[ID0]], %[[ID1]]
